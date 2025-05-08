Setting up  TurboVNC viewer on the server machine to forward the display to the client machine.

# Install TurboVNC
1. Download the TurboVNC viewer from the official website: [TurboVNC Download](https://www.turbovnc.org)

2. As root run the following command to install the downloaded package:
   ```bash
   sudo sh -C "wget -q -O- https://packagecloud.io/dcommander/turbovnc/gpgkey | gpg --dearmor >/etc/apt/trusted.gpg.d/TurboVNC.gpg"
   ```

3. Add TurboVNC list to the system:
    ```bash
    wget https://raw.githubusercontent.com/TurboVNC/repo/main/TurboVNC.list -O /etc/apt/sources.list.d/TurboVNC.list
    ```

4. Update the package list:
    ```bash
    sudo apt update
    ```

5. Install TurboVNC:
    ```bash
    sudo apt install turbovnc
    ```

Setting up  VirtualGL on the server machine to enable OpenGL rendering (GPU acceleration) for TurboVNC.

# Install VirtualGL
1. Download the VirtualGL package from the official website: [VirtualGL Download](https://www.virtualgl.org/)
2. As root run the following command to install the downloaded package:
   ```bash
   sudo sh -C "wget -q -O- https://packagecloud.io/virtualgl/virtualgl/gpgkey | gpg --dearmor >/etc/apt/trusted.gpg.d/VirtualGL.gpg"
   ```
3. Add VirtualGL list to the system:
    ```bash
    wget https://raw.githubusercontent.com/VirtualGL/repo/main/VirtualGL.list -O /etc/apt/sources.list.d/VirtualGL.list
    ```

4. Update the package list:
    ```bash
    sudo apt update
    ```
5. Install VirtualGL:
    ```bash
    sudo apt install virtualgl
    ```

# Configure VirtualGL
1. Run the following command to configure VirtualGL:
   ```bash
   sudo /opt/VirtualGL/bin/vglserver_config
   ```
   This command will prompt you to configure VirtualGL. Follow the instructions to set up VirtualGL for your system.
   Choose option 1 to configure VirtualGL to use both EGL and GLX. This will allow you to use OpenGL rendering with TurboVNC.


# Start TurboVNC server
1. Setup the VNC server to use the desired window manager. Open the VNC server configuration file (if it doesn't exist, create it):
   ```bash
   nano ~/.vnc/xstartup.turbovnc
   ```
   Add the following lines to the file:
   ```bash
   #!/bin/sh
   # Uncomment the following two lines for normal desktop:
   unset SESSION_MANAGER
   unset DBUS_SESSION_BUS_ADDRESS
   # Start the VNC server with the desired window manager
   # Uncomment the following line to use the XFCE window manager:
   exec startxfce4 &
   # Uncomment the following line to use the default window manager:
   # exec /etc/X11/xinit/xinitrc &
   # Uncomment the following line to use the LXDE window manager:
   # exec startlxde &
   # Uncomment the following line to use the MATE window manager:
   # exec mate-session
   # Uncomment the following line to use the KDE window manager:
   # exec startkde &
   # Uncomment the following line to use the GNOME window manager:
   # exec gnome-session &
   # Uncomment the following line to use the Cinnamon window manager:
   # exec cinnamon-session &
   ```
    Save and exit the file.

2. Make the xstartup file executable:
   ```bash
   chmod +x ~/.vnc/xstartup.turbovnc
   ```

3. Start the TurboVNC server with the following command:
   ```bash
   vncserver -d :1 -geometry 1920x1080 -depth 24 
   ```
   This command will start a TurboVNC server on display :1 with a resolution of 1920x1080 and a color depth of 24 bits.
   You can adjust the resolution and color depth according to your needs.

4. Set a password for the VNC server:
   ```bash
    vncpasswd
    ```

5. Check if the TurboVNC server is running:
   ```bash
   vncserver -list
   ```
   This command will list all the running TurboVNC servers. You should see the server you just started in the list.



