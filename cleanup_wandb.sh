#!/bin/bash

# Script to clean up wandb directories without checkpoints or videos
# Also keeps only the last 10 checkpoints in directories that have them
WANDB_DIR="/home/user/cognitiverl/wandb"
DRY_RUN=false  # Set to false to actually delete
MAX_CHECKPOINTS=10  # Maximum number of checkpoints to keep per directory

echo "🔍 Scanning wandb directories for cleanup..."
echo "📁 Directory: $WANDB_DIR"
echo "🔢 Max checkpoints per directory: $MAX_CHECKPOINTS"
echo ""

# Counters
total_dirs=0
dirs_with_checkpoints=0
dirs_with_videos=0
dirs_to_remove=0
total_size_to_remove=0
total_checkpoints_removed=0
total_checkpoint_size_removed=0

# Arrays to store directories
declare -a dirs_to_keep=()
declare -a dirs_to_delete=()ast

# Function to clean up old checkpoints in a directory
cleanup_old_checkpoints() {
    local checkpoints_dir="$1"
    local run_name="$2"
    
    if [[ ! -d "$checkpoints_dir" ]]; then
        return
    fi
    
    # Find all .pt files and sort by step number (extracted from filename)
    local checkpoint_files=()
    while IFS= read -r -d '' file; do
        checkpoint_files+=("$file")
    done < <(find "$checkpoints_dir" -name "ckpt_*.pt" -print0 | sort -z -t_ -k2 -n)
    
    local num_checkpoints=${#checkpoint_files[@]}
    
    if [[ $num_checkpoints -gt $MAX_CHECKPOINTS ]]; then
        local num_to_remove=$((num_checkpoints - MAX_CHECKPOINTS))
        echo "  📦 $run_name has $num_checkpoints checkpoints, removing oldest $num_to_remove"
        
        # Remove the oldest checkpoints (keep the last MAX_CHECKPOINTS)
        for ((i=0; i<num_to_remove; i++)); do
            local file_to_remove="${checkpoint_files[i]}"
            local file_size=$(stat -c%s "$file_to_remove" 2>/dev/null || echo 0)
            total_checkpoint_size_removed=$((total_checkpoint_size_removed + file_size))
            total_checkpoints_removed=$((total_checkpoints_removed + 1))
            
            if [[ "$DRY_RUN" == true ]]; then
                echo "    🗑️  Would remove: $(basename "$file_to_remove")"
            else
                echo "    🗑️  Removing: $(basename "$file_to_remove")"
                rm -f "$file_to_remove"
            fi
        done
    else
        echo "  📦 $run_name has $num_checkpoints checkpoints (≤$MAX_CHECKPOINTS, keeping all)"
    fi
}

# Scan all run directories
for run_dir in "$WANDB_DIR"/run-* "$WANDB_DIR"/offline-run-*; do
    if [[ -d "$run_dir" ]]; then
        total_dirs=$((total_dirs + 1))
        run_name=$(basename "$run_dir")
        
        # Check for checkpoint files
        checkpoints_dir="$run_dir/files/checkpoints"
        has_checkpoints=false
        if [[ -d "$checkpoints_dir" ]] && [[ $(find "$checkpoints_dir" -name "*.pt" 2>/dev/null | wc -l) -gt 0 ]]; then
            has_checkpoints=true
            dirs_with_checkpoints=$((dirs_with_checkpoints + 1))
        fi
        
        # Check for video files
        videos_dir="$run_dir/files/videos/play"
        has_videos=false
        if [[ -d "$videos_dir" ]] && [[ $(find "$videos_dir" -name "*.mp4" 2>/dev/null | wc -l) -gt 0 ]]; then
            has_videos=true
            dirs_with_videos=$((dirs_with_videos + 1))
        fi
        
        # Check for any video files in the entire files/videos directory
        videos_parent_dir="$run_dir/files/videos"
        if [[ -d "$videos_parent_dir" ]] && [[ $(find "$videos_parent_dir" -name "*.mp4" 2>/dev/null | wc -l) -gt 0 ]]; then
            has_videos=true
        fi
        
        # Decide whether to keep or remove
        if [[ "$has_checkpoints" == true ]] || [[ "$has_videos" == true ]]; then
            dirs_to_keep+=("$run_name")
            if [[ "$has_checkpoints" == true ]] && [[ "$has_videos" == true ]]; then
                echo "✅ KEEP: $run_name (has checkpoints AND videos)"
            elif [[ "$has_checkpoints" == true ]]; then
                echo "💾 KEEP: $run_name (has checkpoints)"
            else
                echo "🎬 KEEP: $run_name (has videos)"
            fi
            
            # Clean up old checkpoints in directories we're keeping
            if [[ "$has_checkpoints" == true ]]; then
                cleanup_old_checkpoints "$checkpoints_dir" "$run_name"
            fi
        else
            dirs_to_delete+=("$run_dir")
            dirs_to_remove=$((dirs_to_remove + 1))
            
            # Calculate directory size
            dir_size=$(du -sb "$run_dir" 2>/dev/null | cut -f1)
            total_size_to_remove=$((total_size_to_remove + dir_size))
            
            echo "🗑️  REMOVE: $run_name (no checkpoints or videos)"
        fi
    fi
done

echo ""
echo "📊 SUMMARY:"
echo "   Total directories scanned: $total_dirs"
echo "   Directories with checkpoints: $dirs_with_checkpoints"
echo "   Directories with videos: $dirs_with_videos"
echo "   Directories to keep: $((total_dirs - dirs_to_remove))"
echo "   Directories to remove: $dirs_to_remove"

if [[ $total_size_to_remove -gt 0 ]]; then
    human_size=$(numfmt --to=iec-i --suffix=B $total_size_to_remove)
    echo "   Total size from removed directories: $human_size"
fi

if [[ $total_checkpoints_removed -gt 0 ]]; then
    checkpoint_human_size=$(numfmt --to=iec-i --suffix=B $total_checkpoint_size_removed)
    echo "   Old checkpoints removed: $total_checkpoints_removed files ($checkpoint_human_size)"
fi

total_freed=$((total_size_to_remove + total_checkpoint_size_removed))
if [[ $total_freed -gt 0 ]]; then
    total_human_size=$(numfmt --to=iec-i --suffix=B $total_freed)
    echo "   🎉 Total space to be freed: $total_human_size"
fi

echo ""

# Handle directory removal
if [[ ${#dirs_to_delete[@]} -gt 0 ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo "🔍 DRY RUN MODE - No files will be deleted"
        echo "To actually remove these directories, change DRY_RUN=false in the script"
        echo ""
        echo "Directories that would be removed:"
        for dir in "${dirs_to_delete[@]}"; do
            echo "  - $(basename "$dir")"
        done
    else
        echo "⚠️  DELETING DIRECTORIES..."
        for dir in "${dirs_to_delete[@]}"; do
            echo "Removing: $(basename "$dir")"
            rm -rf "$dir"
        done
        echo "✅ Directory cleanup completed!"
    fi
else
    echo "🎉 No directories need to be removed!"
fi

# Summary of checkpoint cleanup
if [[ $total_checkpoints_removed -gt 0 ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        echo ""
        echo "🔍 Checkpoint cleanup (DRY RUN):"
        echo "   Would remove $total_checkpoints_removed old checkpoint files"
    else
        echo ""
        echo "✅ Checkpoint cleanup completed:"
        echo "   Removed $total_checkpoints_removed old checkpoint files"
    fi
fi

echo ""
echo "📋 Directories being kept:"
for dir in "${dirs_to_keep[@]}"; do
    echo "  - $dir"
done