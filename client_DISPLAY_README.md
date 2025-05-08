Setting up  TurboVNC viewer on the client machine to view the display of the server machine.

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

# Configure TurboVNC

1. Configure the ssh config file to use the TurboVNC viewer. Open the ssh config file:
```bash
nano ~/.ssh/config
```

2. Add the following lines to the file:
```bash
Host display
    HostName $HOST_ADDRESS
    Port $PORT
    User $USER
    IdentityFile $IDENTITY_FILE
    IdentitiesOnly yes
    LocalForward 5901 localhost:5901
```
   Replace `$HOST_ADDRESS`, `$PORT`, `$USER`, and `$IDENTITY_FILE` with the appropriate values for your server.
    - `$HOST_ADDRESS`: The IP address or hostname of the server.
    - `$PORT`: The SSH port of the server (default is 22).
    - `$USER`: The username to log in to the server.
    - `$IDENTITY_FILE`: The path to the private key file used for SSH authentication.

    Save and exit the file.

3. Now run the following command to forward the display and connect to the server:
```bash
ssh -fN display & vncviewer localhost:5901
```
   This command will create an SSH tunnel to the server and forward the display port (5901) to the local machine.

Alternatively, you can use the following command to connect to the server:
```bash
ssh -fN -L 5901:localhost:5901 -p $PORT -i $IDENTITY_FILE $USER@$HOST_ADDRESS & vncviewer localhost:5901
```
   This command will also create an SSH tunnel to the server and forward the display port (5901) to the local machine.
   Replace `$IDENTITY_FILE`, `$USER`, and `$HOST_ADDRESS` with the appropriate values for your server.

The password for the viewer session is 123456 (If needed can be changed later).