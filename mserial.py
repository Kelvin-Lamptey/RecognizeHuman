import serial.tools.list_ports

ports = serial.tools.list_ports.comports()
instance = serial.Serial()
ports_list = []

for port in ports:
	ports_list.append(str(port))
	# print(str(port))

# com = input("Select a port by number: ")
com="3"
for i,port in enumerate(ports_list):
	if ports_list[i].startswith("COM"+ com):
		use = "COM" + com
		# print(use)

instance.baudrate = 9600
instance.port = use
instance.open()

while True:
	command = input("Enter something(1,0,00): ")
	if command == "1":
		a="ON"
		instance.write(a.encode('utf-8'))
	elif command == "0":
		a="OFF"
		instance.write(a.encode('utf-8'))
		pass
	elif command == "00":
		exit()
