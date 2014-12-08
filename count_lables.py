import pdb
from sys import argv

def count_labels(file_name):
	with open (file_name, 'rb') as f:
		zeros = 0
		ones = 0
		for line in f:
			example = line.strip().split(",")
			[example_id, label, hour, c1, banner_pos, site_id, site_domain, site_category, app_id,
			app_domain, app_category, device_id, device_ip, device_model, device_type, device_conn_type,
			c14,c15,c16,c17,c18,c19,c20,c21] = example
			if label == '0':
				zeros += 1
			else:
				ones += 1
		print "num zeros:", zeros, "num ones:", ones

file_name = argv[1]
count_labels(file_name)