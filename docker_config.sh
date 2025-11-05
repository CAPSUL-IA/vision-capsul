git clone https://github.com/jetsonhacks/install-docker
cd install-docker
bash ./install_nvidia_docker.sh
bash ./configure_nvidia_docker.sh
bash ./downgrade_docker.sh
cd .. && rm -rf install-docker

# If there is any error startint the docker daemon run the following code 
# sudo apt reinstall docker-ce
# sudo update-alternatives --set iptables /usr/sbin/iptables-legacy
# sudo update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
# sudo systemctl restart docker
