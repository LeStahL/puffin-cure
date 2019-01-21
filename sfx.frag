/* 
 *  Puffin Cure by Team210 - 64k Demo at Under Construction 2k18
 * 
 *  Copyright (C) 2017  QM <TODO>
 *
 *  This program is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License
 *  as published by the Free Software Foundation; either version 2
 *  of the License, or (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
 
#version 130

uniform float iBlockOffset;
uniform float iSampleRate;
uniform float iVolume;
uniform int iTexS;

#define PI radians(180.)

vec2 mainSound(float t)
{
    return vec2(sin(2.*PI*440.*t));
}

// void main()     
// {
// //     float rtrk = 10./iSampleRate;
//     vec2 uv = gl_FragCoord.xy - .5;
//     float t = (iBlockOffset + dot(uv, vec2(1., 512.)))/iSampleRate;
//     vec2 y = (mainSound( t ));
//     vec2 v  = round(floor((.5+.5*y)*65536.));
//     vec2 vl = mod(v, 255.),
//         vh = floor(v/255.)/255.;
//     gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
// }
// 
void main() 
{
   // compute time `t` based on the pixel we're about to write
   // the 512.0 means the texture is 512 pixels across so it's
   // using a 2 dimensional texture, 512 samples per row
   float t = (iBlockOffset + (gl_FragCoord.x-0.5) + (gl_FragCoord.y-0.5)*float(iTexS))/iSampleRate;
    
//    t = mod(t, 4.5);
    
   // Get the 2 values for left and right channels
   vec2 y = iVolume * mainSound( t );

   // convert them from -1 to 1 to 0 to 65536
   vec2 v  = floor((0.5+0.5*y)*65535.0);

   // separate them into low and high bytes
   vec2 vl = mod(v,256.0)/255.0;
   vec2 vh = mod(floor(v/256.0), 256.)/255.0;

   // write them out where 
   // RED   = channel 0 low byte
   // GREEN = channel 0 high byte
   // BLUE  = channel 1 low byte
   // ALPHA = channel 1 high byte
   gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
}
