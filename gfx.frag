/* Puffin Cure by Team210 - 64k Demo at Under Construction 2018
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
 
uniform float iNBeats;
uniform float iScale;
uniform float iTime;
uniform vec2 iResolution;
uniform sampler2D iFont;
uniform float iFontWidth;

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

vec3 taylorInvSqrt(vec3 r) 
{     
    return 1.79284291400159-0.85373472095314*r; 
}

vec3 permute(vec3 x)
{
    return mod((x*34.+1.)*x, 289.);
}

/* Simplex noise -
Copyright (C) 2011 by Ashima Arts (Simplex noise)
Copyright (C) 2011-2016 by Stefan Gustavson (Classic noise and others)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
float snoise(vec2 P) 
{     
    const vec2 C = vec2 (0.211324865405187134, 0.366025403784438597);  
    vec2 i = floor(P+dot(P, C.yy)) ; 
    vec2 x0 = P-i+dot(i, C.xx) ; 
    // Other  corners 
    vec2 i1 ; 
    i1.x = step ( x0.y , x0.x ) ;  //  1.0  i f  x0 . x > x0 . y ,  e l s e  0.0 
    i1.y = 1.0 - i1.x ; 
    // x1 = x0 − i1 + 1.0 ∗ C. xx ;  x2 = x0 − 1.0 + 2.0 ∗ C. xx ; 
    vec4 x12 = x0.xyxy + vec4 ( C.xx , C.xx * 2.0 - 1.0) ; 
    x12.xy -= i1 ; 
    //  Permutations 
    i = mod( i ,  289.0) ;  // Avoid  truncation  in  polynomial  evaluation 
    vec3 p = permute ( permute ( i.y + vec3 (0.0 , i1.y ,  1.0  ) ) + i.x + vec3 (0.0 , i1.x ,  1.0  ) ) ; 
    //  Circularly  symmetric  blending  kernel
    vec3 m = max(0.5 - vec3 ( dot ( x0 , x0 ) ,  dot ( x12.xy , x12.xy ) , dot ( x12.zw , x12.zw ) ) ,  0.0) ; 
    m = m * m ; 
    m = m * m ; 
    //  Gradients  from 41  points  on a  line ,  mapped onto a diamond 
    vec3 x = fract ( p * (1.0  /  41.0) ) * 2.0 - 1.0  ; 
    vec3 gy = abs ( x ) - 0.5  ; 
    vec3 ox = floor ( x + 0.5) ;  // round (x)  i s  a GLSL 1.30  feature 
    vec3 gx = x - ox ; //  Normalise  gradients  i m p l i c i t l y  by  s c a l i n g m 
    m *= taylorInvSqrt ( gx * gx + gy * gy ) ; // Compute  f i n a l  noise  value  at P 
    vec3 g ; 
    g.x = gx.x * x0.x + gy.x * x0.y ; 
    g.yz = gx.yz * x12.xz + gy.yz * x12.yw ; 
    //  Scale  output  to  span  range  [ − 1 ,1] 
    //  ( s c a l i n g  f a c t o r  determined by  experiments ) 
    return  -1.+2.*(130.0 * dot ( m , g ) ) ; 
}

// TODO: 3D simplex noise

// Multi-frequency simplex noise
float mfsnoise(vec2 x, float f0, float f1, float phi)
{
    float sum = 0.;
    float a = 1.2;
    
    for(float f = f0; f<f1; f = f*2.)
    {
        sum = a*snoise(f*x) + sum;
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

float lineseg(vec3 x, vec3 p1, vec3 p2)
{
    vec3 d = p2-p1;
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

// distance to gear
float dgear(vec2 x, vec2 r, float n)
{
    float p = atan(x.y,x.x);
    p = mod(p, 2.*pi/n)*n/2./pi;
    return mix(length(x)-r.x, length(x)-r.y, step(p,.5));
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
//#define
float dist(vec2 p0,vec2 p1,vec2 p2,vec2 x,float t)
{
    t = clamp(t, 0., 1.);
    return length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);
}
float dist(vec3 p0,vec3 p1,vec3 p2,vec3 x,float t)
{
    t = clamp(t, 0., 1.);
    return length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);
}

// length function, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
#define length23(v) dot(v,v)

//minimum distance to quadratic bezier spline
float spline2(vec2 p0, vec2 p1, vec2 p2, vec2 x)
{
    // check bbox, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
	vec2 bmi = min(p0,min(p1,p2));
    vec2 bma = max(p0,max(p1,p2));
    vec2 bce = (bmi+bma)*0.5;
    vec2 bra = (bma-bmi)*0.5;
    float bdi = length23(max(abs(x-bce)-bra,0.0));
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

//minimum distance to quadratic bezier spline
float spline2(vec3 p0, vec3 p1, vec3 p2, vec3 x)
{
    // check bbox, credits go to IQ / rgba; https://www.shadertoy.com/view/MdyfWc
	vec3 bmi = min(p0,min(p1,p2));
    vec3 bma = max(p0,max(p1,p2));
    vec3 bce = (bmi+bma)*0.5;
    vec3 bra = (bma-bmi)*0.5;
    float bdi = length23(max(abs(x-bce)-bra,0.0));
    if( bdi>dmin )
        return dmin;
        
    //coefficients for 0 = t^3 + a * t^2 + b * t + c
    vec3 E = x-p0, F = p2-2.*p1+p0, G = p1-p0;
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
// extrusion
float zextrude(float z, float d2d, float h)
{
    vec2 w = vec2(-d2d, abs(z)-.5*h);
    return length(max(w,0.));
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

float softmin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float softabs(float x, float a)
{
    return -softmin(x,-x,a);
}

vec2 softabs(vec2 x, float a)
{
	return -vec2(softmin(x.x,-x.x,a), softmin(x.y,-x.y,a));
}

float dtetrahedron(vec3 x, float a, float w)
{
    return abs(softmin(
        lineseg(vec3(softabs(x.x, .5*a),x.yz), c.yyy, a*vec3(1.,0.,-1./sqrt(2.))),
        lineseg(vec3(x.x,softabs(x.y, .5*a),x.z), c.yyy, a*vec3(0.,1.,1./sqrt(2.))),
        .5*a
        ))-w;
}

float dicosahedron(vec3 x, float a, float w)
{
    mat3 r = rot(.3*sin(vec3(1.1,2.2,3.3)*iTime+.5*pi));
    float phi = .5*(1.+sqrt(5.)),
        d = softmin(
            	//lineseg(vec3(x.x, softabs(x.yz,.5*a)), c.yyy, a*vec3(0., 1., phi)),
            	spline2(c.yyy, .5*a*vec3(0., 1., phi), a*r*vec3(0., 1., phi), vec3(x.x, softabs(x.yz,.5*a))),
            	//lineseg(vec3(softabs(x.x, .5*a), x.y, softabs(x.z,.5*a)), c.yyy, a*vec3(phi, 0., 1.)),
            	spline2(c.yyy, .5*a*vec3(phi, 0., 1.), a*r*vec3(phi, 0., 1.), vec3(softabs(x.x, .5*a), x.y, softabs(x.z,.5*a))),
            	.5*a
         	);
    //d = softmin(d, lineseg(vec3(softabs(x.xy, .5*a), x.z), c.yyy, a*vec3(1., phi, 0.)), .5*a);
    d = softmin(d, spline2(c.yyy, .5*a*vec3(1., phi, 0.), r*a*vec3(1., phi, 0.), vec3(softabs(x.xy, .5*a), x.z)), .5*a);
    return abs(d)-w;
}

vec2 talien(vec3 x, float a)
{
    vec3 dt = .01*vec3(sin(iTime), cos(iTime), sin(iTime));
    float dr = .3*a;
//    vec2 sdf = vec2(dtetrahedron(x,.2,.04), 1.);
    vec2 sdf = vec2(dicosahedron(x, .2, .04), 1.);
    vec3 y = mod(x, dr)-.5*dr, 
        ind = (x-y);
    //float da = dtetrahedron(ind, .2, .04);
    float da = dicosahedron(ind, .2, .04);
    if(abs(da)-.025 < 0.)
	    sdf = add(sdf, vec2(length(y)-(.05+.1*rand(ind))*a, 4.));

    // Guards
	float guard = -length(max(abs(y)-vec3(.5*dr*c.xx, .6),0.));
    guard = abs(guard)+dr*.1;
    sdf.x = min(sdf.x, guard);
    
    return sdf;
}

// compute distance to regular triangle
float dtrir(vec2 uv, float r)
{
    float dp = 2.*pi/3.;
    vec2 p0 = vec2(r, 0.),
        p1 = r*vec2(cos(dp), -sin(dp)),
        p2 = r*vec2(cos(2.*dp), -sin(2.*dp)), 
        pd = p2-p1;
    
    float d = min(dot(uv-p0,c.xz*(p1-p0).yx),dot(uv-p1, pd.yx*c.xz));
	return min(d, dot(uv-p2, (p0-p2).yx*c.xz))/length(pd);
}

vec3 ind = c.xxx;

vec2 scene1(vec3 x) // Mountain scene
{
    x += 2.*iTime*c.yxy-.05*x.y;
    
    float dr = .3;
    vec3 y = mod(x, dr)-.5*dr;
    float tlo = clamp(mfsnoise(x.xy, 1.e-1, 5.e-1, .4),-.1,.1), 
        thi = mfsnoise(x.xy, 5.e-1, 5.e2, .4);
    
    // Mountains
    float d = x.z +.2 - .3*(.5*tlo + thi);
    d = min(d, x.z + 1. - .1*thi);
    
    // Guards
    float guard = -length(max(abs(y)-vec3(.5*dr*c.xx, .6),0.));
    guard = abs(guard)+dr*.1;
    d = min(d, guard);

    vec2 sdf = vec2(d, 1.);
   
    // Floor
    vec2 sda = vec2(x.z+1./*-.001*sin(205.*x.y-5.*iTime)*/, 2.);
    sdf = mix(sdf, sda, step(sda.x, sdf.x));
    return sdf;
}

vec2 scene2(vec3 x) // Virus scene
{
    vec2 sdf = c.xy;
    x *= rot(vec3(1.1,2.2,3.3)*iTime);
    sdf = add(sdf,talien(x, .2));
    return sdf;
}

vec2 scene3(vec3 x) // UNC triangle scene
{
    vec2 sdf = c.xy;
    
    x *= rot(vec3(1.1,2.2,3.3)*iTime);
    
    float rb;
    for(int i=0; i<4; ++i)
    {
        rb = .15+.05*rand(float(i+6)*c.xx+7.);
        mat3 r = rot(2.*vec3(1.1,2.2,3.3)*iTime+10.*rand(float(i+13)*c.xx+4.));
        vec3 z = r*(x-rb*(-c.xxx+1.5*vec3(rand(float(i+1)*c.xx), rand(float(i+2)*c.xx), rand(float(i+3)*c.xx))));
        sdf = add(sdf, vec2((zextrude(z.z, dtrir(z.xy, rb), rb)), 1.));
    }
    for(int i=4; i<8; ++i)
    {
        rb = .15+.05*rand(float(i+6)*c.xx+7.);
        mat3 r = rot(2.*vec3(1.1,2.2,3.3)*iTime+10.*rand(float(i+13)*c.xx+4.));
        vec3 z = r*(x-rb*(-c.xxx+1.5*vec3(rand(float(i+1)*c.xx), rand(float(i+2)*c.xx), rand(float(i+3)*c.xx))));
        sdf = add(sdf, vec2((zextrude(z.z, dtrir(z.xy, rb), rb)), 4.));
    }
    return sdf;
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
	
//camera for mountain
void camera1(out vec3 ro, out vec3 r, out vec3 u, out vec3 t)
{
    ro = .5*c.yyx;
    r = c.xyy;
    u = c.yyx+.3*c.yxy;
    t = c.yxy+.4*c.yyx;
}

//camera for virus scene
void camera2(out vec3 ro, out vec3 r, out vec3 u, out vec3 t)
{
    ro = c.yyx;
    r = c.xyy;
    u = c.yxy;
    t = c.yyy;
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

bool hfloor = false;
vec3 color(float rev, float ln, float index, vec2 uv, vec3 x)
{
    vec3 col = c.yyy;
    if(index == 1.)
    {
        x *= 1.e-2;
   		vec3 c1 = stdcolor(1.5e2*x.z+x.xy+.5*rand(ind.xy+17.)/*+iNBeats*/), 
        	c2 = stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)/*+iNBeats*/+11.+uv), 
            c3 = stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)/*+iNBeats*/+23.+uv);
		col = .1*c1*vec3(1.,1.,1.) + .2*c1*vec3(1.,1.,1.)*ln + 1.5*vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.))) + 2.*c1*pow(rev, 8.)+3.*c1*pow(rev, 16.);
        col = clamp(.23*col, 0., 1.);
	}
    else if(index == 2.)
    {
        x *= 1.e-1;
        hfloor = true;
        return .5*stdcolor(x.xy+.5*rand(ind.xy+17.)/*+iNBeats*/);
    }
    else if(index == 3.)
    {
        x *= 1.e-2;
   		vec3 c1 = stdcolor(1.5e2*x.z+x.xy+.5*rand(ind.xy+27.)+iNBeats), 
        	c2 = stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)+iNBeats+21.+uv), 
            c3 = stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)+iNBeats+33.+uv);
		col = .4*c1*vec3(1.,1.,1.) + .2*c1*vec3(1.,1.,1.)*ln + .5*vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.))) + 2.*c1*pow(rev, 8.)+3.*c1*pow(rev, 16.);
        col = clamp(col, 0., 1.);
    }
    else if(index == 4.)
    {
        x *= 1.e-2;
   		vec3 c1 = 1.*stdcolor(1.5e2*x.z+x.xy+.5*rand(ind.xy+47.)/*+iNBeats*/+14.), 
        	c2 = 1.*stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)/*+iNBeats*/+21.+uv), 
            c3 = 1.*stdcolor(1.5e2*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)/*+iNBeats*/+33.+uv);
		col = .1*c1*vec3(1.,1.,1.) + .2*c1*vec3(1.,1.,1.)*ln + 1.5*vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.))) + 2.*c1*pow(rev, 8.)+3.*c1*pow(rev, 16.);
        col = clamp(.23*col, 0., 1.);
	}
    return col;
}

// Add thickness effect to object
vec4 thick(vec2 x, vec4 sdf, vec2 n)
{
    for(int i=1; i<6; ++i)
		sdf = add(vec4(stroke(sdf.x*n.x*n.y*2.*snoise((3.+4.*iScale)*x-2.-1.*iTime-1.2), .01), 3.e-3/abs(sdf.x+.2*snoise(x-2.-1.*iTime))*stdcolor(x+c.xx*.3*float(i))), sdf); 
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

float star(vec2 x, float r0)
{
    return 1.-smoothstep(.5*r0, r0, length(x));
}

vec3 bandc(vec2 x, float a)
{
    return mix(c.yyy, c.xxy, step(.5*a, mod(x.x+x.y-.1*iTime, a)));
}

vec4 gir(vec2 x, float r)
{
    vec4 sdf = vec4(dgear(x, vec2(r-.015, r), floor(107.143*r)), c.xxy);
    sdf = add(sdf, vec4(length(x)-.536*r, c.yyy));
    sdf = add(sdf, vec4(abs(length(x)-.321*r)-.036*r, c.xxy));
    return sdf;
}

vec3 background1(vec2 x)
{
    //Stars
    float dr = .03, scale;
    vec2 y = mod(x, dr)-.5*dr;
    float rs = rand(x-y)*.005,
        dx = -.5*(dr-rs)+(dr-2.*rs)*rand(x-y+1.),
        dy = -.5*(dr-rs)+(dr-2.*rs)*rand(x-y+2.);
    scale = star(y-vec2(dx,dy), rs);
    vec3 color = scale*clamp(8.*rand(x.xy+4.)*stdcolor(rand(x-y+3.)*x.xy), 0., 1.); 
    
    // Star nebula
    float f = mfsnoise(x.xy-6.93, 2.e-1, 1.e2, .55);
    color += mix(c.yyy, stdcolor(x), .5+.95*f);
    color += mix(c.yyy, 2.*stdcolor(x+4.), .5+.33*f);
    color += mix(c.yyy, stdcolor(x+8.), .5+.79*f);
    
    return clamp(color, 0., 1.);
}

// Background for the unc logo
vec3 background2(vec2 x)
{
    //x *= rot((-1.+snoise(.1*iTime*c.xx))*.65*iTime);
    vec3 bg = c.yyy;
    float p = atan(x.y,x.x)/iTime,
        n = 5.,
        dmax = .3+.1*snoise(iTime*c.xx);
    for(float i = 0.; i<n; i+=1.)
    {
        float d = i/n*dmax;
        bg += background1((length(x)-.05+d-2.*iTime)*vec2(cos(p), sin(p))-.05*vec2(snoise(x.xy-iTime), snoise(x.xy+iTime)));
    }
    bg /= n;
    return bg;
}

vec2 spd(vec3 x)
{
    return vec2(length(x)-.45, 1.);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.yy-.5;
    vec3 col = c.yyy;
    
    // Scene 1: 2D; Greet the party.
    if(iTime < 6.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // "Hello, UNC 2018"
        {
            size = 1.54;
            carriage = -.2*c.xy;
            int str[14] = int[14](72, 101, 108, 108, 111, 32, 85, 78, 67, 32, 50, 48, 49, 56);
            for(int i=0; i<14; ++i)
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(1., 5., 1.);        
    }
    
    // Scene 2: 2D; "Loading Bar" Logo with gears
    else if(iTime < 16.)
    {
       	col = mix(clamp(col,c.yyy,c.xxx), bandc(uv, .1), smoothstep(1.5/iResolution.y, -1.5/iResolution.y, stroke(logo(uv-.5*c.xy,.2),.05)));
       	col = mix(clamp(col,c.yyy,c.xxx), c.xxy, smoothstep(1.5/iResolution.y, -1.5/iResolution.y, stroke(stroke(logo(uv-.5*c.xy,.2),.05),.001)));
    	
        float n = 15., dr = .8*.14;
		mat2 r = rot(1.1*iTime), mr = rot(-1.1*iTime-2.*pi/n*dr);
        vec4 sdf = gir(r*(uv-.3*c.xy), dr);
        sdf = add(sdf, gir(mr*(uv-.09*c.xy), dr));
        sdf = add(sdf, gir(r*(uv-.7*c.xy), dr));
        
        //decoration gears
        sdf = add(sdf, gir(r*r*(uv+.065*c.xy), .5*dr));
        sdf = add(sdf, gir(mr*(uv+.22*c.xy), dr));
        
        col = mix(clamp(col,c.yyy,c.xxx), sdf.gba, smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x))* blend(7., 15., 1.);
    }
    
    // Scene 3: Say who did this.
    else if(iTime < 22.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // "QM. NR4. Team210."
        {
            size = 1.54;
            carriage = -.35*c.xy;
            int str[17] = int[17](81, 77, 46, 32, 78, 82, 52, 46, 32, 84, 101, 97, 109, 50, 49, 48, 46);
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(17., 21., 1.);        
    }
    
    // Scene 4: Mountains.
    else if(iTime < 42.)
    {
        vec3 ro, r, u, t, x, dir;
    	camerasetup(camera1, ro, r, u, t, uv, dir);
    	
        float d = (.15-ro.z)/dir.z;
        if(uv.y>.1)//THAT WAS THE RELEVANT OPTIMIZATION
        {
            // Draw Background here.
            col = background1(uv);
             
            post(col, uv);
            fragColor = vec4(col * blend(23., 41., 1.), 1.);
            return;
        }
        else
        {
            bool hit;
            vec2 s;
            raymarch(scene1, x, ro, d, dir, s, 300, 1.e-4, hit);
            if(hit == false || x.y > 12.)
            {
                // Draw Background here.
                col = background1(uv);
                
                post(col, uv);
                fragColor = vec4(col * blend(23., 41., 1.), 1.);
                return;
            }

            vec3 n;
            calcnormal(scene1, n, 5.e-3, x);

            vec3 l = x+2.*c.yyx, re = normalize(reflect(-l,n)), v = normalize(x-ro);
            float rev = abs(dot(re,v)), ln = abs(dot(l,n));

            col = color(rev, ln, s.y, uv, x);

            if(s.y == 2.)
            {

                for(float i = .7; i > .5; i -= .2)
                {
                    //reflections
                    dir = normalize(reflect(dir, n));
        //             dir = normalize(refract(dir, n, i));
                    d = 5.e-2;
                    ro = x;
                    raymarch(scene1, x, ro, d, dir, s, 50, 1.e-3, hit);
                    if(hit == false||x.y>12.)
                    {
                        col = mix(col,background1(uv), .5);
                        post(col, uv);
                        fragColor = vec4(col, 1.);
                        break;
                    }
                    calcnormal(scene1, n, 1.e-3, x);
                    l = x+2.*c.yyx;
                    re = normalize(reflect(-l,n)); 
                    v = normalize(x-ro);
                    rev = abs(dot(re,v));
                    ln = abs(dot(l,n));

                    col = mix(col, color(rev, ln, s.y, uv, x), i);
                }
            }
        }
        
        col *=  blend(23., 41., 1.);
    }
    
    // Greet the friends.
    else if(iTime < 48.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = c.yy;//1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // See GREETINGS.
        if(uv.x<1.)
        {
            size = 1.04;
            int str[24];
            int nstr;
            if(uv.y > .335+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy+.4*c.yx;
                str = int[24](68, 101, 107, 97, 100, 101, 110, 99, 101, 46, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0);
            }
            else if(uv.y > .235+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy+.3*c.yx;
                str = int[24](74, 117, 109, 97, 108, 97, 117, 116, 97, 46, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0);
            }
            else if(uv.y > .135+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy+.2*c.yx;
                str = int[24](75, 101, 119, 108, 101, 114, 115, 32, 38, 32, 77, 70, 88, 46, 0, 0, 0,0,0,0,0,0,0,0);
            }
            else if(uv.y > .035+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy+.1*c.yx;
                str = int[24](68, 97, 109, 111, 110, 101, 115, 46, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0);
            }
            else if(uv.y > -.065+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy+.0*c.yx;
                str = int[24](77, 97, 116, 116, 32, 67, 117, 114, 114, 101, 110, 116, 46, 0, 0, 0, 0,0,0,0,0,0,0,0);
            }
            else if(uv.y > -.3+vn.y && uv.x < 1.)
            {
                carriage = -.4*c.xy-.15*c.yx;
                if(mod(floor(2.*iTime),2.) == 0.)
                {
                    str = int[24](83, 99, 104, 110, 97, 112, 112, 115, 103, 105, 114, 108, 115, 0,0,0,0,0,0,0,0,0,0,0);
                }
                else
                {
                    str = int[24](83, 99, 104, 110, 97, 114, 99, 104, 103, 105, 114, 108, 115, 0,0,0,0,0,0,0,0,0,0,0);
                }
            }
            else
            {
                col = c.yyy;
                post(col, uv);
                fragColor = vec4(col, 1.);
                return;
            }
            for(int i=0; i<24; ++i)
            {
                //if( (abs(uv.x) < 1.5) && (abs(uv.y) < .1) )
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(43., 47., 1.) ;
    }
    
    // Alien virus scene
    else if(iTime < 68.)
    {
        vec3 ro, r, u, t, x, dir;
    	camerasetup(camera2, ro, r, u, t, uv, dir);
    	
        float d = 0.;//(.15-ro.z)/dir.z;
        //if(d<2.)
        if(uv.x > .5)
        {
            // Draw Background here.
                col = background1(uv) * blend(49., 67., 1.);
                
                post(col, uv);
                fragColor = vec4(col, 1.);
                return;
        }
        else
        {
            bool hit;
            vec2 s;
            raymarch(scene2, x, ro, d, dir, s, 200, 1.e-4, hit);
            if(hit == false || x.y > 12.)
            {
                // Draw Background here.
                col = background1(uv) * blend(49., 67., 1.);
                
                post(col, uv);
                fragColor = vec4(col, 1.);
                return;
            }

            vec3 n;
            calcnormal(scene2, n, 5.e-3, x);

            vec3 l = x+2.*c.yyx, re = normalize(reflect(-l,n)), v = normalize(x-ro);
            float rev = abs(dot(re,v)), ln = abs(dot(l,n));

            col = color(rev, ln, s.y, uv, x) * blend(49., 67., 1.);
        }
    }
    
    // New year wishes
    else if(iTime < 74.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // "Happy 2019 to you!"
        {
            size = 1.54;
            carriage = -.35*c.xy;
            int str[18] = int[18](72, 97, 112, 112, 121, 32, 50, 48, 49, 57, 32, 116, 111, 32, 121, 111, 117, 33);
            for(int i=0; i<18; ++i)
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(69., 73., 1.);        
    }
    
    // Under Construction logo
    else if(iTime < 95.)
    {
        vec3 ro, r, u, t, x, dir;
        vec2 s;
    	camerasetup(camera2, ro, r, u, t, uv, dir);
    	
        // Graph traversal
        float d = 0.;
        bool hit0;
        raymarch(spd, x, ro, d, dir, s, 15, 1.e-2, hit0);
        
        if(!hit0)
        {
            // Draw Background here.
            col = background2(uv)* blend(76., 94., 1.);
                
            post(col, uv);
            fragColor = vec4(col, 1.);
            return;
        }
        else
        {
            bool hit;
            
            raymarch(scene3, x, ro, d, dir, s, 150, 1.e-4, hit);
            if(hit == false || x.y > 12.)
            {
                // Draw Background here.
                col = background2(uv)* blend(76., 94., 1.);
                
                post(col, uv);
                fragColor = vec4(col, 1.);
                return;
            }
            
            vec3 n;
            calcnormal(scene3, n, 5.e-3, x);

            vec3 l = x+2.*c.yyx, re = normalize(reflect(-l,n)), v = normalize(x-ro);
            float rev = abs(dot(re,v)), ln = abs(dot(l,n));

            col = color(rev, ln, s.y, uv, x) * blend(76., 94., 1.);
        }
    }
    
    // We go to revision!
    else if(iTime < 101.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // "See you at Revision."
        {
            size = 1.54;
            carriage = -.4*c.xy;
            int str[20] = int[20](83, 101, 101, 32, 121, 111, 117, 32, 97, 116, 32, 82, 101, 118, 105, 115, 105, 111, 110, 46);
            for(int i=0; i<20; ++i)
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(96., 100., 1.);        
    }
    
    // We are Team210!
    else if(iTime < 107.)
    {
        vec4 sdf = vec4(1., col);
        float d = 1., dc = 1., dca = 1.;
        
        vec2 vn = 1.e-2*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime), snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));
        
        // "See you at Revision."
        {
            size = 1.34;
            carriage = -.45*c.xy;
            int str[22] = int[22](84, 101, 97, 109, 50, 49, 48, 46, 32, 119, 119, 119, 46, 122, 49, 48, 46, 105, 110, 102, 111, 32);
            for(int i=0; i<22; ++i)
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
        d = stroke(d, 3.4e-3)+.1*length(vn);
        sdf = add(sdf, vec4(d, c.xxx));
        sdf = add(sdf, vec4(dca, c.xxx));
        sdf = add(sdf, vec4(dc, c.xxy));
        
        col = sdf.gba * smoothstep(1.5/iResolution.y, -1.5/iResolution.y, sdf.x) * blend(102., 106., 1.);        
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

