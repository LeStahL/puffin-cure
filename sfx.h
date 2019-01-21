/* File generated with Shader Minifier 1.1.5
 * http://www.ctrl-alt-test.fr
 */
#ifndef SFX_H_
# define SFX_H_

const char *sfx_frag =
 "#version 130\n"
 "uniform float iBlockOffset,iSampleRate,iVolume;"
 "uniform int iTexS;\n"
 "#define PI radians(180.)\n"
 "vec2 i(float i)"
 "{"
   "return vec2(sin(2.*PI*440.*i));"
 "}"
 "void main()"
 "{"
   "float f=(iBlockOffset+(gl_FragCoord.x-.5)+(gl_FragCoord.y-.5)*float(iTexS))/iSampleRate;"
   "vec2 r=iVolume*i(f),g=floor((.5+.5*r)*65535.),d=mod(g,256.)/255.,o=mod(floor(g/256.),256.)/255.;"
   "gl_FragColor=vec4(d.x,o.x,d.y,o.y);"
 "}";

#endif // SFX_H_
