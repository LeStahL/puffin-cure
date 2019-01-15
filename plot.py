import matplotlib.pyplot as plt
import struct
import numpy

data = None
with open('SOUND', 'rb') as f:
    data = f.read()

print("Length of data is: ", len(data))

x = []
y = []
for i in range(400):
    t = float(i)/44100.
    x += [ struct.unpack_from('e', data, 2*i)[0] ]
    if i < 400.:
        y += [ .1*numpy.floor(.5+.5*numpy.sin(2.*numpy.pi*440.0*t)*65536.) ]

fig = plt.figure()
plt.plot(range(len(x[:400])),x[:400], 'ro-')
#plt.plot(range(len(x[:400])),y[:400], 'go-')
plt.show()

input('Press any Key...')
