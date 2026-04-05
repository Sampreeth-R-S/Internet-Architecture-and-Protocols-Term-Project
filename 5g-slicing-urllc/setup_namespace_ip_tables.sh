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
for i in $(seq 0 10); do
	table_id=$((200 + i))
	echo "$table_id ue$i" | sudo tee -a /etc/iproute2/rt_tables
done
for i in $(seq 0 10); do
	sudo ip route add 10.45.0.0/16 dev "uesimtun$i" table "ue$i"
done
for i in $(seq 0 10); do
	ue_ip_octet=$((2 + i))
	sudo ip rule add from "10.45.0.$ue_ip_octet" table "ue$i"
done
for i in $(seq 0 10); do
	sudo ip route replace 10.45.0.0/16 dev "uesimtun$i" table "ue$i"
done
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
for i in $(seq 0 10); do
	sudo ip route replace 10.46.0.0/24 dev "uesimtun$i" table "ue$i"
done
ip rule
ip addr show ogstun
sudo ip route flush cache
ip rule
ip route show table ue1
sudo ip route flush cache
ip route get 10.46.0.10 from 10.45.0.3
for i in $(seq 0 10); do
	ue_ip_octet=$((2 + i))
	priority=$((100 + i))
	sudo ip rule add from "10.45.0.$ue_ip_octet" table "ue$i" priority "$priority"
done
sudo ip route flush cache
ip rule
ip route get 10.46.0.10 from 10.45.0.3
sudo ip netns exec dn ip route add 10.45.0.0/16 dev ogstun
