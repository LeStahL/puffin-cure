# Star Wave Rave by Team210 - 64k Demo at Vortex III 2k18
# Copyright (C) 2018  Alexander Kraus <nr4@z10.info>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse # For command line args

# Parse command line arguments
parser = argparse.ArgumentParser(description='Team210 shader minifier.')
parser.add_argument('-o', '--output', dest='out')
parser.add_argument('-0', '--no-minification', dest='no_minification', action='store_true')
args, rest = parser.parse_known_args()

# Exit if inconsistent argument combinations are supplied
if rest == []:
    print("No input present. Doing nothing. Type 'shader_minifier -h' for help.")
    exit()

# Write to stdout if no output file is specified
print_stdout = False
write_c_header = False
write_fragment_shader = False
guard = None
if args.out == None:
    print_stdout = True
else:
    # Determine output format from args.out
    dots = args.out.split('\\')[-1].split('/')[-1].split('.')
    extension = dots[-1]
    if extension == 'h':
        # Output to C header
        write_c_header = True
        # Determine include guard ID from rest
        guard = '_'.join(dots).upper()
    elif extension == 'frag':
        # Output to fragment shader
        write_fragment_shader = True
        
# Read input
content = None
with open(rest[0], 'rt') as f:
    content = f.read()
    f.close()

# Array that gets saved to the output file
output = ""

# Perform no minification and save shader to header
if args.no_minification != None:
    if args.no_minification and write_c_header:
        output += '/* Generated with shader-compressor by NR4/Team210. */\n'
        output += '#ifndef ' + guard + '\n'
        output += '#define ' + guard + '\n'
        output += 'const char * ' + guard[:-1].lower() + 'frag =\n'
        
        for line in content.split('\n'):
            line = line.replace('"', '\\\"')
            output += '"' + line + '\\n"\n'
        
        output += ';\n'
        output += '#endif\n'
            

# Save output
if print_stdout:
    print(output)
elif write_c_header:
    print(args.out)
    with open(args.out, 'wt') as f:
        f.write(output)
        f.close()
        
        
