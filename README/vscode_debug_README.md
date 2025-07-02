# VSCode Setup for Isaac Sim

1. VSCode launch configuration
    Place the following code in `.vscode/launch.json` file in your workspace:

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Isaac Sim",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "args": []
            }
        ]
    }
    ```

2. VSCode settings
    Place the following code in `.vscode/settings.json` file in your workspace:

    ```json
    {
        "python.defaultInterpreterPath": "${env:HOME}/isaaclab_env/bin/python",
        "python.terminal.activateEnvironment": true,
        "python.analysis.extraPaths": [
            "${workspaceFolder}"
        ]
    }
    ```

3. Configure Ruff extension setup in user settings.json
    Place the following code in user settings.json
    ```json
    {
        "[python]": {
            "editor.formatOnSave": true,
            "editor.codeActionsOnSave": {
                "source.fixAll": "explicit",
                "source.organizeImports": "explicit"
            },
            "editor.defaultFormatter": "charliermarsh.ruff"
        },
        "notebook.formatOnSave.enabled": true,
        "notebook.codeActionsOnSave": {
            "notebook.source.fixAll": "explicit",
            "notebook.source.organizeImports": "explicit"
        }
    }
    ```