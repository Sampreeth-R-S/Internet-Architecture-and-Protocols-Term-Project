ip addr show ogstun
sudo ip addr add 10.45.0.10/16 dev ogstun
ip addr show ogstun
sudo sysctl -w net.ipv4.conf.all.rp_filter=0
sudo sysctl -w net.ipv4.conf.ogstun.rp_filter=0
sudo iptables -F
sudo ip route add 10.45.0.0/16 dev uesimtun0
echo "200 uesim" | sudo tee -a /etc/iproute2/rt_tables
sudo ip route add 10.45.0.0/16 dev uesimtun0 table uesim
sudo ip rule add from 10.45.0.2 table uesim
ip rule
ip route show table uesim
echo "200 ue0" | sudo tee -a /etc/iproute2/rt_tables
echo "201 ue1" | sudo tee -a /etc/iproute2/rt_tables
echo "202 ue2" | sudo tee -a /etc/iproute2/rt_tables
echo "203 ue3" | sudo tee -a /etc/iproute2/rt_tables
echo "204 ue4" | sudo tee -a /etc/iproute2/rt_tables
sudo ip route add 10.45.0.0/16 dev uesimtun0 table ue0
sudo ip route add 10.45.0.0/16 dev uesimtun1 table ue1
sudo ip route add 10.45.0.0/16 dev uesimtun2 table ue2
sudo ip route add 10.45.0.0/16 dev uesimtun3 table ue3
sudo ip route add 10.45.0.0/16 dev uesimtun4 table ue4
sudo ip rule add from 10.45.0.2 table ue0
sudo ip rule add from 10.45.0.3 table ue1
sudo ip rule add from 10.45.0.4 table ue2
sudo ip rule add from 10.45.0.5 table ue3
sudo ip rule add from 10.45.0.6 table ue4
sudo ip route replace 10.45.0.0/16 dev uesimtun0 table ue0
sudo ip route replace 10.45.0.0/16 dev uesimtun1 table ue1
sudo ip route replace 10.45.0.0/16 dev uesimtun2 table ue2
sudo ip route replace 10.45.0.0/16 dev uesimtun3 table ue3
sudo ip route replace 10.45.0.0/16 dev uesimtun4 table ue4
ip route show table ue0
ip rule
ip route get 10.45.0.10 from 10.45.0.3
sudo ip addr add 10.46.0.10/24 dev ogstun
ifconfig
ip addr show
sudo ip route add 10.46.0.0/24 dev ogstun
ip route get 10.46.0.10 from 10.45.0.3
ip route | grep 10.46
ip route get 10.46.0.10 from 10.45.0.3
sudo ip netns add dn
sudo ip link set ogstun netns dn
sudo ip netns exec dn ip addr add 10.46.0.10/24 dev ogstun
sudo ip netns exec dn ip link set ogstun up
ip route get 10.46.0.10 from 10.45.0.3
sudo ip route replace 10.46.0.0/24 dev uesimtun0 table ue0
sudo ip route replace 10.46.0.0/24 dev uesimtun1 table ue1
sudo ip route replace 10.46.0.0/24 dev uesimtun2 table ue2
sudo ip route replace 10.46.0.0/24 dev uesimtun3 table ue3
sudo ip route replace 10.46.0.0/24 dev uesimtun4 table ue4
ip rule
ip addr show ogstun
sudo ip route flush cache
ip rule
ip route show table ue1
sudo ip rule add from 10.45.0.3 table ue1 priority 100
sudo ip route flush cache
ip route get 10.46.0.10 from 10.45.0.3
sudo ip rule add from 10.45.0.2 table ue0 priority 100
sudo ip rule add from 10.45.0.3 table ue1 priority 101
sudo ip rule add from 10.45.0.4 table ue2 priority 102
sudo ip rule add from 10.45.0.5 table ue3 priority 103
sudo ip rule add from 10.45.0.6 table ue4 priority 104
sudo ip route flush cache
ip rule
ip route get 10.46.0.10 from 10.45.0.3
sudo ip netns exec dn ip route add 10.45.0.0/16 dev ogstun
