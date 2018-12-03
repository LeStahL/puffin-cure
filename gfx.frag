/* Puffin Cure by Team210 - 64k Demo at Under Construction 2k18
 * Copyright (C) 2018  Alexander Kraus <nr4@z10.info>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#version 130

// Uniforms
uniform float iTime;
uniform vec2 iResolution;
uniform sampler2D iFont;
uniform float iFontWidth;
uniform float iNBeats;
uniform float iScale;

// Global constants
const vec3 c = vec3(1.,0.,-1.);
const float pi = acos(-1.);

// Global variables
float size = 1.,
    dmin = 1.;
vec2 carriage = c.yy, 
    glyphsize = c.yy;
vec3 col = c.yyy;

// Hash function
float rand(vec2 x)
{
    return fract(sin(dot(x-1. ,vec2(12.9898,78.233)))*43758.5453);
}

float rand(vec3 x)
{
    return fract(sin(dot(x-1. ,vec3(12.9898,78.233,33.1818)))*43758.5453);
}

vec3 rand3(vec3 x)
{
    return vec3(rand(x.x*c.xx),rand(x.y*c.xx),rand(x.z*c.xx));
}

mat3 rot(vec3 p)
{
    return mat3(c.xyyy, cos(p.x), sin(p.x), 0., -sin(p.x), cos(p.x))
        *mat3(cos(p.y), 0., -sin(p.y), c.yxy, sin(p.y), 0., cos(p.y))
        *mat3(cos(p.z), -sin(p.z), 0., sin(p.z), cos(p.z), c.yyyx);
}

/* compute voronoi distance and closest point.
 * x: coordinate
 * return value: vec3(distance, coordinate of control point)
 */
vec4 vor(vec3 x)
{
    vec3 y = floor(x);
   	float ret = 10.;
    
    //find closest control point. ("In which cell am I?")
    vec3 pf=c.yyy, p;
    float df=100., d;
    
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
            for(int k=-1; k<=1; k+=1)
            {
                p = y + vec3(float(i), float(j), float(k));
                p += rand3(p);

                d = length(x-p);
				
                if(d < df)
                {
                    df = d;
                    pf = p;
                }
            }
    
    //compute voronoi distance: minimum distance to any edge
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
            for(int k=-1; k<=1; k+=1)
            {
                p = y + vec3(float(i), float(j), float(k));
                p += rand3(p);

                vec3 o = p - pf;
                d = abs(.5-dot(x-pf, o)/length(o));
                ret = min(ret, d);
            }
    return vec4(ret, pf);
}

/* compute voronoi distance and closest point.
 * x: coordinate
 * return value: vec3(distance, coordinate of control point)
 */
vec3 vor(vec2 x)
{
    vec2 y = floor(x);
   	float ret = 1.;
    
    //find closest control point. ("In which cell am I?")
    vec2 pf=c.yy, p;
    float df=10., d;
    
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += vec2(rand(p), rand(p+1.));
            
            d = length(x-p);
            
            if(d < df)
            {
                df = d;
                pf = p;
            }
        }
    
    //compute voronoi distance: minimum distance to any edge
    for(int i=-1; i<=1; i+=1)
        for(int j=-1; j<=1; j+=1)
        {
            p = y + vec2(float(i), float(j));
            p += vec2(rand(p), rand(p+1.));
            
            vec2 o = p - pf;
            d = length(.5*o-dot(x-pf, o)/dot(o,o)*o);
            ret = min(ret, d);
        }
    
    return vec3(ret, pf);
}

// 2D value noise
float valuenoise(vec2 x)
{
    vec2 y = floor(x);
    x = fract(x);
    float r00 = -1.+2.*rand(y),
        r10 = -1.+2.*rand(y+c.xy),
        r01 = -1.+2.*rand(y+c.yx),
        r11 = -1.+2.*rand(y+c.xx);
    return mix(
        mix(r00, r10, x.x),
        mix(r01, r11, x.x),
        x.y
    );
}

// 3D value noise
float valuenoise(vec3 x)
{
    vec3 y = floor(x);
    x = fract(x);
    float r000 = -1.+2.*rand(y),
        r100 = -1.+2.*rand(y+c.xyy),
        r010 = -1.+2.*rand(y+c.yxy),
        r001 = -1.+2.*rand(y+c.yyx),
        r110 = -1.+2.*rand(y+c.xxy),
        r011 = -1.+2.*rand(y+c.yxx),
        r101 = -1.+2.*rand(y+c.xyx),
        r111 = -1.+2.*rand(y+c.xxx);
    return 	mix(
        		mix(
            		mix(r000, r100, smoothstep(0.,1.,x.x)),
                    mix(r010, r110, smoothstep(0.,1.,x.x)),
                    smoothstep(0.,1.,x.y)
                ),
        		mix(
                    mix(r001, r101, smoothstep(0.,1.,x.x)),
                    mix(r011, r111, smoothstep(0.,1.,x.x)),
                    smoothstep(0.,1.,x.y)
                ),
        		smoothstep(0.,1.,x.z));
        
}

// Multi-frequency value noise
float mfvaluenoise(vec2 x, float f0, float f1, float phi)
{
    float sum = 0.;
    float a = 1.2;
    
    for(float f = f0; f<f1; f = f*2.)
    {
        sum = a*valuenoise(f*x) + sum;
        a = a*phi;
    }
    
    return sum;
}
    
// Add objects to scene with proper antialiasing
vec4 add(vec4 sdf, vec4 sda)
{
    return vec4(
        min(sdf.x, sda.x), 
        mix(sda.gba, sdf.gba, smoothstep(-1.5/iResolution.y, 1.5/iResolution.y, sda.x))
    );
}

// add object to scene
vec2 add(vec2 sda, vec2 sdb)
{
    return mix(sda, sdb, step(sdb.x, sda.x));
}

// subtract object from scene
vec2 sub(vec2 sda, vec2 sdb)
{
    return mix(-sda, sdb, step(sda.x, sdb.x));
}

// Add objects to scene with blending
vec4 smoothadd(vec4 sdf, vec4 sda, float a)
{
    return vec4(
        min(sdf.x, sda.x), 
        mix(sda.gba, sdf.gba, smoothstep(-a*1.5/iResolution.y, a*1.5/iResolution.y, sda.x))
    );
}

// Distance to line segment
float lineseg(vec2 x, vec2 p1, vec2 p2)
{
    vec2 d = p2-p1;
    return length(x-mix(p1, p2, clamp(dot(x-p1, d)/dot(d,d),0.,1.)));
}

// distance to spiral
float dspiral(vec2 x, float a, float d)
{
    float p = atan(x.y, x.x),
        n = floor((abs(length(x)-a*p)+d*p)/(2.*pi*a));
    p += (n*2.+1.)*pi;
    return -abs(length(x)-a*p)+d*p;
}

// Distance to circle
float circle(vec2 x, float r)
{
    return length(x)-r;
}

// Distance to circle segment
float circlesegment(vec2 x, float r, float p0, float p1)
{
    float p = atan(x.y, x.x);
    p = clamp(p, p0, p1);
    return length(x-r*vec2(cos(p), sin(p)));
}

// Distance to 210 logo
float logo(vec2 x, float r)
{
    return min(
        min(circle(x+r*c.zy, r), lineseg(x,r*c.yz, r*c.yx)),
        circlesegment(x+r*c.xy, r, -.5*pi, .5*pi)
    );
}

// Distance to stroke for any object
float stroke(float d, float w)
{
    return abs(d)-w;
}

//distance to quadratic bezier spline with parameter t
float dist(vec2 p0,vec2 p1,vec2 p2,vec2 x,float t)
{
    t = clamp(t, 0., 1.);
    return length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);
}

// length function, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
float length2( in vec2 v  ) { return dot(v,v); }

//minimum distance to quadratic bezier spline
float spline2(vec2 p0, vec2 p1, vec2 p2, vec2 x)
{
    // check bbox, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
	vec2 bmi = min(p0,min(p1,p2));
    vec2 bma = max(p0,max(p1,p2));
    vec2 bce = (bmi+bma)*0.5;
    vec2 bra = (bma-bmi)*0.5;
    float bdi = length2(max(abs(x-bce)-bra,0.0));
    if( bdi>dmin )
        return dmin;
        
    //coefficients for 0 = t^3 + a * t^2 + b * t + c
    vec2 E = x-p0, F = p2-2.*p1+p0, G = p1-p0;
    vec3 ai = vec3(3.*dot(G,F), 2.*dot(G,G)-dot(E,F), -dot(E,G))/dot(F,F);

	//discriminant and helpers
    float tau = ai.x/3., p = ai.y-tau*ai.x, q = - tau*(tau*tau+p)+ai.z, dis = q*q/4.+p*p*p/27.;
    
    //triple real root
    if(dis > 0.) 
    {
        vec2 ki = -.5*q*c.xx+sqrt(dis)*c.xz, ui = sign(ki)*pow(abs(ki), c.xx/3.);
        return dist(p0,p1,p2,x,ui.x+ui.y-tau);
    }
    
    //three distinct real roots
    float fac = sqrt(-4./3.*p), arg = acos(-.5*q*sqrt(-27./p/p/p))/3.;
    vec3 t = c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;
    return min(
        dist(p0,p1,p2,x, t.x),
        min(
            dist(p0,p1,p2,x,t.y),
            dist(p0,p1,p2,x,t.z)
        )
    );
}

// extrusion
float zextrude(float z, float d2d, float h)
{
    vec2 d = abs(vec2(min(d2d, 0.),z))-h*c.yx;
    return min(max(d.x,d.y),0.)+length(max(d,0.));
}

// Read short value from texture at index off
float rshort(float off)
{
    // Parity of offset determines which byte is required.
    float hilo = mod(off, 2.);
    // Find the pixel offset your data is in (2 unsigned shorts per pixel).
    off *= .5;
    // - Determine texture coordinates.
    //     offset = i*iFontWidth+j for (i,j) in [0,iFontWidth]^2
    //     floor(offset/iFontWidth) = floor((i*iFontwidth+j)/iFontwidth)
    //                              = floor(i)+floor(j/iFontWidth) = i
    //     mod(offset, iFontWidth) = mod(i*iFontWidth + j, iFontWidth) = j
    // - For texture coordinates (i,j) has to be rescaled to [0,1].
    // - Also we need to add an extra small offset to the texture coordinate
    //   in order to always "hit" the right pixel. Pixel width is
    //     1./iFontWidth.
    //   Half of it is in the center of the pixel.
    vec2 ind = (vec2(mod(off, iFontWidth), floor(off/iFontWidth))+.05)/iFontWidth;
    // Get 4 bytes of data from the texture
    vec4 block = texture(iFont, ind);
    // Select the appropriate word
    vec2 data = mix(block.rg, block.ba, hilo);
    // Convert bytes to unsigned short. The lower bytes operate on 255,
    // the higher bytes operate on 65280, which is the maximum range 
    // of 65535 minus the lower 255.
    return round(dot(vec2(255., 65280.), data));
}

// Compute distance to glyph from ascii value out of the font texture.
// This function parses glyph point and control data and computes the correct
// Spline control points. Then it uses the signed distance function to
// piecewise bezier splines to get a signed distance to the font glyph.
float dglyph(vec2 x, int ascii)
{
    // Treat spaces
    if(ascii == 32)
    {
        glyphsize = size*vec2(.02,1.);
        return 1.;
    }

    // Get glyph index length
    float nchars = rshort(0.);
    
    // Find character in glyph index
    float off = -1.;
    for(float i=0.; i<nchars; i+=1.)
    {
        int ord = int(rshort(1.+2.*i));
        if(ord == ascii)
        {
            off = rshort(1.+2.*i+1);
            break;
        }
    }
    // Ignore characters that are not present in the glyph index.
    if(off == -1.) return 1.;
    
    // Get short range offsets. Sign is read separately.
    vec2 dx = mix(c.xx,c.zz,vec2(rshort(off), rshort(off+2.)))*vec2(rshort(off+1.), rshort(off+3.));
    
    // Read the glyph splines from the texture
    float npts = rshort(off+4.),
        xoff = off+5., 
        yoff = off+6.+npts,
        toff = off+7.+2.*npts, 
        coff = off+8.+3.*npts,
        ncont = rshort(coff-1.),
        d = 1.;
    
    // Save glyph size
    vec2 mx = -100.*c.xx,
        mn = 100.*c.xx;
    
    // Loop through the contours of the glyph. All of them are closed.
    for(float i=0.; i<ncont; i+=1.)
    {
        // Get the contour start and end indices from the contour array.
        float istart = 0., 
            iend = rshort(coff+i);
        if(i>0.)
            istart = rshort(coff+i-1.) + 1.;
        
        // Prepare a stack
        vec2 stack[3];
        float tstack[3];
        int stacksize = 0;
        
        // Loop through the segments
        for(float j = istart; j <= iend; j += 1.)
        {
            tstack[stacksize] = rshort(toff + j);
            stack[stacksize] = (vec2(rshort(xoff+j), rshort(yoff+j)) + dx)/65536.*size;
            mx = max(mx, stack[stacksize]);
            mn = min(mn, stack[stacksize]);
            ++stacksize;
            
            // Check if line segment is finished
            if(stacksize == 2)
            {
                if(tstack[0]*tstack[1] == 1)
                {
                    d = min(d, lineseg(x, stack[0], stack[1]));
                    --j;
                    stacksize = 0;
                }
            }
            else 
            if(stacksize == 3)
            {
                if(tstack[0]*tstack[2] == 1.)
                {
                    d = min(d, spline2(stack[0], stack[1], stack[2], x));
                    --j;
                    stacksize = 0;
                }
                else
                {
                    vec2 p = mix(stack[1], stack[2], .5);
                    d = min(d, spline2(stack[0], stack[1], p, x));
                    stack[0] = p;
                    tstack[0] = 1.;
                    mx = max(mx, stack[0]);
                    mn = min(mn, stack[0]);
                    --j;
                    stacksize = 1;
                }
            }
        }
        tstack[stacksize] = rshort(toff + istart);
        stack[stacksize] = (vec2(rshort(xoff+istart), rshort(yoff+istart)) + dx)/65536.*size;
        mx = max(mx, stack[0]);
        mn = min(mn, stack[0]); 
        ++stacksize;
        if(stacksize == 2)
        {
            d = min(d, lineseg(x, stack[0], stack[1]));
        }
        else 
        if(stacksize == 3)
        {
            d = min(d, (spline2(stack[0], stack[1], stack[2], x)));
        }
    }
    
    glyphsize = abs(mx-mn);
    
    return d;
}

// Compute distance to glyph control points for debug purposes
float dglyphpts(vec2 x, int ascii)
{
    // Get glyph index length
    float nchars = rshort(0.);
    
    // Find character in glyph index
    float off = -1.;
    for(float i=0.; i<nchars; i+=1.)
    {
        int ord = int(rshort(1.+2.*i));
        if(ord == ascii)
        {
            off = rshort(1.+2.*i+1);
            break;
        }
    }
    // Ignore characters that are not present in the glyph index.
    if(off == -1.) return 1.;
    
    // Get short range offsets. Sign is read separately.
    vec2 dx = mix(c.xx,c.zz,vec2(rshort(off), rshort(off+2.)))*vec2(rshort(off+1.), rshort(off+3.));
    
    // Read the glyph splines from the texture
    float npts = rshort(off+4.),
        xoff = off+5., 
        yoff = off+6.+npts,
        d = 1.;
        
    // Debug output of the spline control points
    for(float i=0.; i<npts; i+=1.)
    {
        vec2 xa = ( vec2(rshort(xoff+i), rshort(yoff+i)) + dx )/65536.*size;
        d = min(d, length(x-xa)-2.e-3);
    }
    
    return d;
}

// Two-dimensional rotation matrix
mat2 rot(float t)
{
    vec2 sc = vec2(cos(t), sin(t));
    return mat2(sc*c.xz, sc.yx);
}

float blend(float tstart, float tend, float dt)
{
    return smoothstep(tstart-dt, tstart + dt, iTime)*(1.-smoothstep(tend-dt, tend+dt, iTime));
}

vec3 ind;
vec2 scene(vec3 x) // Vortex scene
{
    x += iTime*c.yxy*1.e-1-iNBeats*c.xxy;
    
    vec2 dis = 12.*vec2((.1+.05*iScale)*valuenoise(x-2.-2.e-1*iTime),(.1+.05*iScale)*valuenoise(x.xy-5.-2.e-1*iTime));
    float d = x.z - mfvaluenoise(x.xy-dis, 2., 40., .45+.2*clamp(3.*iScale, 0., 1.));
    d = max(d, -.5*mfvaluenoise(x.xy-dis, 2., 10., .45+.2*clamp(3.*iScale,0., 1.)));
   
    float dr = .165;
    vec3 y = mod(x, dr)-.5*dr;
    float guard = -length(max(abs(y)-vec3(.5*dr*c.xx, .6),0.));
    guard = abs(guard)+dr*.1;
    d = min(d, guard);
    
    return vec2(d, 1.);
}

vec2 scene2(vec3 x) // City scene
{
    //displacement
    vec2 dis = 12.*vec2((.1+.05*iScale)*valuenoise(x-2.-1.e-1*iTime),(.1+.05*iScale)*valuenoise(x.xy-5.-1.e-1*iTime));
    
    //city
    vec3 v = vor(7.*x.xy-iNBeats-dis);
//     float h = .3*clamp(iScale, 0., 1.)*valuenoise(v.yz+1.);
    float h = .4*rand(v.yz+7.+iNBeats)+.2*valuenoise(v.yz+7.+iNBeats)*clamp(5.*iScale, 0., 1.)+.1*valuenoise(v.yz-.4*iTime);
    float d = stroke(zextrude(x.z+.1-.2*x.y, stroke(v.x,.15+.15*clamp(3.*iScale, 0., 1.)), h),.1);

    //artificial guards for artifacts
    float dr = .065;
    vec3 y = mod(x, dr)-.5*dr;
    float guard = -length(max(abs(y)-vec3(.5*dr*c.xx, .6),0.));
    guard = abs(guard)+dr*.1;
    d = min(d, guard);
    
    //coloring
    ind = vec3(v.yz, 0.);

    //floor
    return add(vec2(d, 1.), vec2(x.z, 2.));
}

vec2 scene3(vec3 x) // shattered sphere glass
{
    x = rot(.05*vec3(1.,2.,3.)*iTime+iNBeats*c.xxx)*x;
    vec3 y = mod(x, 1.)-.5;
    vec4 v = (length(y)-.5)*c.xxxx, w  = vor(2.*x-(.2+.1*iScale)*valuenoise(x.xy-2.-1.*iTime));
    ind = v.gba+.1*w.gba;
    float d = max(-stroke(.4*w.x,5.e-3+1.e-3*iScale), stroke(v.x,1.e-1));
    d = max(-length(x)+1., d);
    return vec2(abs(d)-.001, 1.);
}

vec2 scene4(vec3 x)
{
    x = rot(.05*vec3(1.,2.,3.)*iTime+iNBeats)*x;
    vec4 v = vor(.5*x-(.1+.05*iScale)*valuenoise(x.xy-2.-1.*iTime)), w  = vor(2.*x-(.2+.1*iScale)*valuenoise(x.xy-2.-1.*iTime));
    ind = v.gba+.1*w.gba;
    float d = max(-stroke(.4*w.x,5.e-3+1.e-3*iScale), stroke(v.x,1.e-1));
    //d = max(-length(x)+1., d);
    return vec2(abs(d)-.001, 1.);
}

//performs raymarching
//scene: name of the scene function
//xc: 	 name of the coordinate variable
//ro:	 name of the ray origin variable
//d:	 name of the distance variable
//dir:	 name of the direction variable
//s:	 name of the scenestruct variable
//N:	 number of iterations used
//eps:	 exit criterion
//flag:  name of the flag to set if raymarching succeeded
#define raymarch(scene, xc, ro, d, dir, s, N, eps, flag) \
	flag = false;\
	for(int i=0; i<N; ++i)\
    {\
        xc = ro + d*dir;\
        s = scene(xc);\
        if(s.x < eps)\
        {\
            flag = true;\
            break;\
        }\
        d += s.x;\
    }

//computes normal with finite differences
//scene: name of the scene function
//n:	 name of the normal variable
//eps:	 precision of the computation
//xc:	 location of normal evaluation
#define calcnormal(scene, n, eps, xc) \
	{\
        float ss = scene(xc).x;\
        n = normalize(vec3(scene(xc+eps*c.xyy).xc-ss,\
                           scene(xc+eps*c.yxy).xc-ss,\
                           scene(xc+eps*c.yyx).xc-ss));\
    }

//camera setup
//camera: camera function with camera(out vec3 ro, out vec3 r, out vec3 u, out vec3 t)
//ro:	  name of the ray origin variable
//r:	  name of the right variable
//u:	  name of the up variable
//t:	  name of the target variable
//uv:	  fragment coordinate
//dir:	  name of the dir variable
#define camerasetup(camera, ro, r, u, t, uv, dir) \
	{\
        camera(ro, r, u, t);\
        t += uv.x*r+uv.y*u;\
        dir = normalize(t-ro);\
    }

//post processing: 210 logo and trendy display lines
//col: output color
//uv:  fragment coordinate
#define post(color, uv) \
	{\
    	col = mix(clamp(col,c.yyy,c.xxx), c.xxx, smoothstep(1.5/iResolution.y, -1.5/iResolution.y, stroke(logo(uv-vec2(-.45,.45),.02),.005)));\
    	col += vec3(0., 0.05, 0.1)*sin(uv.y*1050.+ 5.*iTime);\
	}
	
//camera for scene 1
void camera1(out vec3 ro, out vec3 r, out vec3 u, out vec3 t)
{
    ro = c.yyx;
    r = c.xyy;
    u = c.yxx;
    t = c.yxy;
}

vec3 synthcol(float scale, float phase)
{
    vec3 c2 = vec3(207.,30.,102.)/255.,
        c3 = vec3(245., 194., 87.)/255.;
    mat3 r1 = rot((5.e-1*phase)*vec3(1.1,1.3,1.5));
    return 
        (
            1.1*mix
            (
                -(cross(c2, r1*c2)),
                -(r1*c2), 
                scale
            )
        );
}

vec3 stdcolor(vec2 x)
{
	return 0.5 + 0.5*cos(iTime+x.xyx+vec3(0,2,4));
}

vec3 color(float rev, float ln, float index, vec2 uv, vec3 x)
{
    vec3 col = c.yyy;
    if(index == 1.)
    {
   		vec3 c1 = stdcolor(x.xy+.5*rand(ind.xy+17.)+iNBeats), 
        	c2 = stdcolor(x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)+iNBeats+11.+uv), 
            c3 = stdcolor(x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)+iNBeats+23.+uv);
		col = .1*c1*vec3(1.,1.,1.) + .2*c1*vec3(1.,1.,1.)*ln + vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.))) + 2.*c1*pow(rev, 8.)+3.*c1*pow(rev, 16.);
		col = clamp(.33*col, 0., 1.);
        //col = abs(col);
	}
    else if(index == 2.)
    {
        return stdcolor(x.xy+.5*rand(ind.xy+17.)+iNBeats);
    }
    return col;
}

// Add thickness effect to object
vec4 thick(vec2 x, vec4 sdf, vec2 n)
{
    for(int i=1; i<6; ++i)
		sdf = add(vec4(stroke(sdf.x*n.x*n.y*2.*valuenoise((3.+4.*iScale)*x-2.-1.*iTime-1.2), .01), 3.e-3/abs(sdf.x+.2*valuenoise(x-2.-1.*iTime))*stdcolor(x+c.xx*.3*float(i))), sdf); 
    return sdf;
}

// Draw Geometry
vec4 geometry(vec2 x)
{
    vec4 sdf = vec4(stroke(stroke(logo(x, .2), .06),.01), 2.5*stdcolor(x*1.7));
    //for(int i=0; i<10; ++i)
    //    sdf = add(sdf, vec4(stroke(circle(x-.5*vec2(valuenoise(x-2.-5.*iTime+2.*rand(float(i+3)*c.xx)), valuenoise(x-2.-5.*iTime+rand(float(i)*c.xx)))-.5*c.xy, .2+valuenoise(x-2.-5.*iTime+rand(float(i)*c.xx))),.01), 2.5*stdcolor(x+float(i)*.1)));
    return sdf;
}

// Normal
const float dx = 1.e-4;
vec2 normal(vec2 x)
{
    float s = geometry(x).x;
    return normalize(vec2(geometry(x+dx*c.xy).x-s, geometry(x+dx*c.yx).x-s));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.yy-.5;
    
    // Scene 1: 2D; Greet the party.
    if(iTime < 6.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 2.e-2*vec2(valuenoise(2.3*uv-2.*vec2(1.5,2.4)*iTime), valuenoise(1.7*uv-2.4*vec2(1.2,2.1)*iTime));
        
        // "Hello, Vortex III"
        {
            size = 1.54;
            carriage = -.25*c.xy;
            int str[17] = int[17](72, 101, 108, 108, 111, 44, 32, 86, 111, 114, 116, 101, 120, 32, 73, 73, 73);
            for(int i=0; i<17; ++i)
            {
                if( (abs(uv.x) < 1.5) && (abs(uv.y) < .1) )
                {
                    vec2 bound = uv-carriage-vn+.05*c.yx;
                    d = min(d, dglyph(bound, str[i]));
                    float d0 = dglyphpts(bound, str[i]);
                    dc = min(dc, d0);
                    dca = min(dca, stroke(d0, 2.e-3));
                    carriage += glyphsize.x*c.xy + .01*c.xy;
                }
            }
        }
        d = stroke(d, 2.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xyy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(1., 5., 1.);        
    }
    
    // Post-process
    post(col, uv);
    
    // Set the fragment color
    fragColor = vec4(col, 1.);    
}

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
