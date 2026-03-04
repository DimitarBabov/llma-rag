# Remove any existing port forwarding for this port
netsh interface portproxy delete v4tov4 listenport=5001 listenaddress=192.168.4.74

# Add the new port forwarding
netsh interface portproxy add v4tov4 listenport=5001 listenaddress=192.168.4.74 connectport=5001 connectaddress=172.21.159.120

# Add firewall rules (will show error if they already exist, but that's ok)
netsh advfirewall firewall add rule name="Flask Server In" dir=in action=allow protocol=TCP localport=5001
netsh advfirewall firewall add rule name="Flask Server Out" dir=out action=allow protocol=TCP localport=5001

Write-Host "Port forwarding and firewall rules have been set up!"
Write-Host "You can verify the port forwarding with: netsh interface portproxy show all" 