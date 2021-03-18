import numpy as np
import struct

def readnek(fname):
	"""
	Utility function to read nek5000 binary files.

	Parameters
	----------
	fname : string
    	Name of the file to be read.

	Returns
	-------
	type
    	Description of returned object.

	"""

	####################################
	#####     Parse the header     #####
	####################################

	# --> Open the file.
	try:
	    infile = open(fname, "rb")
	except IOError as e:
	    print("I/O erro({0}): {1}".format(e.errno, e.strerror))
	    return -1

	# --> Read header.
	header = infile.read(132).split()

	# --> Get word size.
	wdsz = int(header[1])
	if wdsz == 4:
	    realtype = "f"
	elif wdsz == 8:
	    readltype = "d"
	else:
	    print("ERROR: Could not interpret real type (wdsz = %i)" %wdsz)
	    return -2

	# --> Get polynomial order.
	lr1 = [int(i) for i in header[2:5]]

	# --> Compute total number of points per element.
	npel = np.prod(lr1)

	# --> Number of physical dimensions.
	ndim = 2 + (lr1[2]>1)

	# --> Get the number of elements.
	nel = int(header[5])

	# --> Get the number of elements in that file.
	nelf = int(header[6])

	# --> Get current time.
	time = float(header[7])

	# --> Get current time-step.
	istep = int(header[8])

	# --> Get file ID.
	fid = int(header[9])

	# --> Get total number of files.
	nf = int(header[10])

	# --> Get variables [XUPT]
	vars = header[11].decode("utf-8")
	var = np.zeros(5, dtype=np.int)
	for v in vars:
	    if v == "X":
	        var[0] = ndim
	    elif v == "U":
	        var[1] = ndim
	    elif v == "P":
	        var[2] = 1
	    elif v == "T":
	        var[3] = 1
	    elif v == "S":
	    	v[4] = 0

	# --> Total number of scalar fields to be read.
	nfields = var.sum()

	# --> Identify endian encoding.
	etagb = infile.read(4)
	etagL = struct.unpack('<f', etagb)[0]
	etagL = int(etagL*1e5)/1e5
	etagB = struct.unpack('>f', etagb)[0]
	etagB = int(etagB*1e5)/1e5

	if etagL == 6.54321:
		emode = '<'
	elif etagB == 6.54321:
		emode = '>'
	else:
		print('ERROR: could not interpret endianness')
		return -3

	# --> Read the element map.
	elmap = infile.read(4*nelf)
	elmap = list(struct.unpack(emode+nelf*"i", elmap))

	########################################
	#####     Read the actual data     #####
	########################################

	# --> Numpy array container for the data.
	data = np.zeros((nelf, npel, nfields))

	for ivar in range(len(var)):
		idim0 = sum(var[:ivar])
		for iel in elmap:
			for idim in range(var[ivar]):
				x = infile.read(npel*wdsz)
				data[iel-1, :, idim+idim0] = struct.unpack(emode+npel*realtype, x)

	# --> Close the file.
	infile.close()

	# --> Output dictionnary.
	output = {
		"data": data,
		"lr1": lr1,
		"elmap": elmap,
		"time": time,
		"istep": istep,
		"emode": emode,
		"wdsz": wdsz,
		"header": header,
		"fields": vars
	}

	return output

def writenek(fname, dico):
	##################################
	#####     Initialization     #####
	##################################

	# --> Open the file.
	try:
		outfile = open(fname, "wb")
	except IOError as e:
		print("I/O erro ({0}) : {1}".format(e.errno, e.strerror))
		return -1

	# --> Misc.
	fid, nf = 0, 1

	# --> Polynomial order.
	lr1 = dico["lr1"]

	# --> Get data.
	data = dico["data"]

	# --> Get element map.
	elmap = dico["elmap"]

	# --> Number of elements, points per element.
	nel, npel, _ = data.shape
	assert npel == np.prod(lr1)

	# --> Number of active dimensions.
	ndim = 2 + (lr1[2]>1)

	# --> Get fields to be written.
	var = np.zeros(5, dtype=np.int)
	for v in dico["fields"]:
	    if v == "X":
	        var[0] = ndim
	    elif v == "U":
	        var[1] = ndim
	    elif v == "P":
	        var[2] = 1
	    elif v == "T":
	        var[3] = 1
	    elif v == "S":
	    	v[4] = 0

	# --> Get word size.
	if dico["wdsz"] == 4:
		realtype = "f"
	elif dico["wdsz"] == 8:
		realtype = "d"
	else:
	    print("ERROR: Could not interpret real type (wdsz = %i)" %wdsz)
	    return -2

	# --> Get endianness.
	emode = dico["emode"]

	# --> Generate header.
	header = "#std %1i %2i %2i %2i %10i %10i %20.13E %9i %6i %6i %s\n" %(
		dico["wdsz"], lr1[0], lr1[1], lr1[2], nel, nel,
		dico["time"], dico["istep"], fid, nf, dico["fields"])

	header = header.ljust(132).encode("utf-8")

	# --> Write header.
	outfile.write(header)

	etagb = struct.pack(emode+"f", 6.54321)
	outfile.write(etagb)

	outfile.write(struct.pack(emode+nel*"i", *elmap))

	##############################
	#####     Write data     #####
	##############################

	for ivar in range(len(var)):
		idim0 = np.sum(var[:ivar])
		for iel in elmap:
			for idim in range(var[ivar]):
				x = struct.pack(emode+npel*realtype, *data[iel-1, :, idim+idim0])
				outfile.write(x)

	# --> Close the file.
	outfile.close()
