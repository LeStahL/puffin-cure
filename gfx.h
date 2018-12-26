/* File generated with Shader Minifier 1.1.5
 * http://www.ctrl-alt-test.fr
 */
#ifndef GFX_H_
# define GFX_H_

const char *gfx_frag =
 "#version 130\n"
 "uniform float iNBeats,iScale,iTime;"
 "uniform vec2 iResolution;"
 "uniform sampler2D iFont;"
 "uniform float iFontWidth;"
 "const vec3 c=vec3(1.,0.,-1.);"
 "const float pi=acos(-1.);"
 "float size=1.,dmin=1.;"
 "vec2 carriage=c.yy,glyphsize=c.yy;"
 "vec3 col=c.yyy;"
 "float rand(vec2 x)"
 "{"
   "return fract(sin(dot(x-1.,vec2(12.9898,78.233)))*43758.5);"
 "}"
 "float rand(vec3 x)"
 "{"
   "return fract(sin(dot(x-1.,vec3(12.9898,78.233,33.1818)))*43758.5);"
 "}"
 "vec3 rand3(vec3 x)"
 "{"
   "return vec3(rand(x.x*c.xx),rand(x.y*c.xx),rand(x.z*c.xx));"
 "}"
 "mat3 rot(vec3 p)"
 "{"
   "return mat3(c.xyyy,cos(p.x),sin(p.x),0.,-sin(p.x),cos(p.x))*mat3(cos(p.y),0.,-sin(p.y),c.yxy,sin(p.y),0.,cos(p.y))*mat3(cos(p.z),-sin(p.z),0.,sin(p.z),cos(p.z),c.yyyx);"
 "}"
 "vec3 vor(vec2 x)"
 "{"
   "vec2 y=floor(x);"
   "float ret=1.;"
   "vec2 pf=c.yy,p;"
   "float df=10.,d;"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "{"
         "p=y+vec2(float(i),float(j));"
         "p+=vec2(rand(p),rand(p+1.));"
         "d=length(x-p);"
         "if(d<df)"
           "df=d,pf=p;"
       "}"
   "for(int i=-1;i<=1;i+=1)"
     "for(int j=-1;j<=1;j+=1)"
       "{"
         "p=y+vec2(float(i),float(j));"
         "p+=vec2(rand(p),rand(p+1.));"
         "vec2 o=p-pf;"
         "d=length(.5*o-dot(x-pf,o)/dot(o,o)*o);"
         "ret=min(ret,d);"
       "}"
   "return vec3(ret,pf);"
 "}"
 "vec3 taylorInvSqrt(vec3 r)"
 "{"
   "return 1.79284-.853735*r;"
 "}"
 "vec3 permute(vec3 x)"
 "{"
   "return mod((x*34.+1.)*x,289.);"
 "}"
 "float snoise(vec2 P)"
 "{"
   "const vec2 C=vec2(.211325,.366025);"
   "vec2 i=floor(P+dot(P,C.yy)),x0=P-i+dot(i,C.xx),i1;"
   "i1.x=step(x0.y,x0.x);"
   "i1.y=1.-i1.x;"
   "vec4 x12=x0.xyxy+vec4(C.xx,C.xx*2.-1.);"
   "x12.xy-=i1;"
   "i=mod(i,289.);"
   "vec3 p=permute(permute(i.y+vec3(0.,i1.y,1.))+i.x+vec3(0.,i1.x,1.)),m=max(.5-vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);"
   "m=m*m;"
   "m=m*m;"
   "vec3 x=fract(p*(1./41.))*2.-1.,gy=abs(x)-.5,ox=floor(x+.5),gx=x-ox;"
   "m*=taylorInvSqrt(gx*gx+gy*gy);"
   "vec3 g;"
   "g.x=gx.x*x0.x+gy.x*x0.y;"
   "g.yz=gx.yz*x12.xz+gy.yz*x12.yw;"
   "return-1.+2.*(130.*dot(m,g));"
 "}"
 "float mfsnoise(vec2 x,float f0,float f1,float phi)"
 "{"
   "float sum=0.,a=1.2;"
   "for(float f=f0;f<f1;f=f*2.)"
     "sum=a*snoise(f*x)+sum,a=a*phi;"
   "return sum;"
 "}"
 "vec4 add(vec4 sdf,vec4 sda)"
 "{"
   "return vec4(min(sdf.x,sda.x),mix(sda.yzw,sdf.yzw,smoothstep(-1.5/iResolution.y,1.5/iResolution.y,sda.x)));"
 "}"
 "vec2 add(vec2 sda,vec2 sdb)"
 "{"
   "return mix(sda,sdb,step(sdb.x,sda.x));"
 "}"
 "vec2 sub(vec2 sda,vec2 sdb)"
 "{"
   "return mix(-sda,sdb,step(sda.x,sdb.x));"
 "}"
 "vec4 smoothadd(vec4 sdf,vec4 sda,float a)"
 "{"
   "return vec4(min(sdf.x,sda.x),mix(sda.yzw,sdf.yzw,smoothstep(-a*1.5/iResolution.y,a*1.5/iResolution.y,sda.x)));"
 "}"
 "float lineseg(vec2 x,vec2 p1,vec2 p2)"
 "{"
   "vec2 d=p2-p1;"
   "return length(x-mix(p1,p2,clamp(dot(x-p1,d)/dot(d,d),0.,1.)));"
 "}"
 "float lineseg(vec3 x,vec3 p1,vec3 p2)"
 "{"
   "vec3 d=p2-p1;"
   "return length(x-mix(p1,p2,clamp(dot(x-p1,d)/dot(d,d),0.,1.)));"
 "}"
 "float dspiral(vec2 x,float a,float d)"
 "{"
   "float p=atan(x.y,x.x),n=floor((abs(length(x)-a*p)+d*p)/(2.*pi*a));"
   "p+=(n*2.+1.)*pi;"
   "return-abs(length(x)-a*p)+d*p;"
 "}"
 "float dgear(vec2 x,vec2 r,float n)"
 "{"
   "float p=atan(x.y,x.x);"
   "p=mod(p,2.*pi/n)*n/2./pi;"
   "return mix(length(x)-r.x,length(x)-r.y,step(p,.5));"
 "}"
 "float circle(vec2 x,float r)"
 "{"
   "return length(x)-r;"
 "}"
 "float circlesegment(vec2 x,float r,float p0,float p1)"
 "{"
   "float p=atan(x.y,x.x);"
   "p=clamp(p,p0,p1);"
   "return length(x-r*vec2(cos(p),sin(p)));"
 "}"
 "float logo(vec2 x,float r)"
 "{"
   "return min(min(circle(x+r*c.zy,r),lineseg(x,r*c.yz,r*c.yx)),circlesegment(x+r*c.xy,r,-.5*pi,.5*pi));"
 "}"
 "float stroke(float d,float w)"
 "{"
   "return abs(d)-w;"
 "}"
 "float dist(vec2 p0,vec2 p1,vec2 p2,vec2 x,float t)"
 "{"
   "return t=clamp(t,0.,1.),length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);"
 "}"
 "float dist(vec3 p0,vec3 p1,vec3 p2,vec3 x,float t)"
 "{"
   "return t=clamp(t,0.,1.),length(x-pow(1.-t,2.)*p0-2.*(1.-t)*t*p1-t*t*p2);"
 "}\n"
 "#define length23(v)dot(v,v)\n"
 "float spline2(vec2 p0,vec2 p1,vec2 p2,vec2 x)"
 "{"
   "vec2 bmi=min(p0,min(p1,p2)),bma=max(p0,max(p1,p2)),bce=(bmi+bma)*.5,bra=(bma-bmi)*.5;"
   "float bdi=length23(max(abs(x-bce)-bra,0.));"
   "if(bdi>dmin)"
     "return dmin;"
   "vec2 E=x-p0,F=p2-2.*p1+p0,G=p1-p0;"
   "vec3 ai=vec3(3.*dot(G,F),2.*dot(G,G)-dot(E,F),-dot(E,G))/dot(F,F);"
   "float tau=ai.x/3.,p=ai.y-tau*ai.x,q=-tau*(tau*tau+p)+ai.z,dis=q*q/4.+p*p*p/27.;"
   "if(dis>0.)"
     "{"
       "vec2 ki=-.5*q*c.xx+sqrt(dis)*c.xz,ui=sign(ki)*pow(abs(ki),c.xx/3.);"
       "return dist(p0,p1,p2,x,ui.x+ui.y-tau);"
     "}"
   "float fac=sqrt(-4./3.*p),arg=acos(-.5*q*sqrt(-27./p/p/p))/3.;"
   "vec3 t=c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;"
   "return min(dist(p0,p1,p2,x,t.x),min(dist(p0,p1,p2,x,t.y),dist(p0,p1,p2,x,t.z)));"
 "}"
 "float spline2(vec3 p0,vec3 p1,vec3 p2,vec3 x)"
 "{"
   "vec3 bmi=min(p0,min(p1,p2)),bma=max(p0,max(p1,p2)),bce=(bmi+bma)*.5,bra=(bma-bmi)*.5;"
   "float bdi=length23(max(abs(x-bce)-bra,0.));"
   "if(bdi>dmin)"
     "return dmin;"
   "vec3 E=x-p0,F=p2-2.*p1+p0,G=p1-p0,ai=vec3(3.*dot(G,F),2.*dot(G,G)-dot(E,F),-dot(E,G))/dot(F,F);"
   "float tau=ai.x/3.,p=ai.y-tau*ai.x,q=-tau*(tau*tau+p)+ai.z,dis=q*q/4.+p*p*p/27.;"
   "if(dis>0.)"
     "{"
       "vec2 ki=-.5*q*c.xx+sqrt(dis)*c.xz,ui=sign(ki)*pow(abs(ki),c.xx/3.);"
       "return dist(p0,p1,p2,x,ui.x+ui.y-tau);"
     "}"
   "float fac=sqrt(-4./3.*p),arg=acos(-.5*q*sqrt(-27./p/p/p))/3.;"
   "vec3 t=c.zxz*fac*cos(arg*c.xxx+c*pi/3.)-tau;"
   "return min(dist(p0,p1,p2,x,t.x),min(dist(p0,p1,p2,x,t.y),dist(p0,p1,p2,x,t.z)));"
 "}"
 "float zextrude(float z,float d2d,float h)"
 "{"
   "vec2 d=abs(vec2(min(d2d,0.),z))-h*c.yx;"
   "return min(max(d.x,d.y),0.)+length(max(d,0.));"
 "}"
 "float rshort(float off)"
 "{"
   "float hilo=mod(off,2.);"
   "off*=.5;"
   "vec2 ind=(vec2(mod(off,iFontWidth),floor(off/iFontWidth))+.05)/iFontWidth;"
   "vec4 block=texture(iFont,ind);"
   "vec2 data=mix(block.xy,block.zw,hilo);"
   "return round(dot(vec2(255.,65280.),data));"
 "}"
 "float dglyph(vec2 x,int ascii)"
 "{"
   "if(ascii==32)"
     "return glyphsize=size*vec2(.02,1.),1.;"
   "float nchars=rshort(0.),off=-1.;"
   "for(float i=0.;i<nchars;i+=1.)"
     "{"
       "int ord=int(rshort(1.+2.*i));"
       "if(ord==ascii)"
         "{"
           "off=rshort(1.+2.*i+1);"
           "break;"
         "}"
     "}"
   "if(off==-1.)"
     "return 1.;"
   "vec2 dx=mix(c.xx,c.zz,vec2(rshort(off),rshort(off+2.)))*vec2(rshort(off+1.),rshort(off+3.));"
   "float npts=rshort(off+4.),xoff=off+5.,yoff=off+6.+npts,toff=off+7.+2.*npts,coff=off+8.+3.*npts,ncont=rshort(coff-1.),d=1.;"
   "vec2 mx=-100.*c.xx,mn=100.*c.xx;"
   "for(float i=0.;i<ncont;i+=1.)"
     "{"
       "float istart=0.,iend=rshort(coff+i);"
       "if(i>0.)"
         "istart=rshort(coff+i-1.)+1.;"
       "vec2 stack[3];"
       "float tstack[3];"
       "int stacksize=0;"
       "for(float j=istart;j<=iend;j+=1.)"
         "{"
           "tstack[stacksize]=rshort(toff+j);"
           "stack[stacksize]=(vec2(rshort(xoff+j),rshort(yoff+j))+dx)/65536.*size;"
           "mx=max(mx,stack[stacksize]);"
           "mn=min(mn,stack[stacksize]);"
           "++stacksize;"
           "if(stacksize==2)"
             "{"
               "if(tstack[0]*tstack[1]==1)"
                 "d=min(d,lineseg(x,stack[0],stack[1])),--j,stacksize=0;"
             "}"
           "else"
             " if(stacksize==3)"
               "{"
                 "if(tstack[0]*tstack[2]==1.)"
                   "d=min(d,spline2(stack[0],stack[1],stack[2],x)),--j,stacksize=0;"
                 "else"
                   "{"
                     "vec2 p=mix(stack[1],stack[2],.5);"
                     "d=min(d,spline2(stack[0],stack[1],p,x));"
                     "stack[0]=p;"
                     "tstack[0]=1.;"
                     "mx=max(mx,stack[0]);"
                     "mn=min(mn,stack[0]);"
                     "--j;"
                     "stacksize=1;"
                   "}"
               "}"
         "}"
       "tstack[stacksize]=rshort(toff+istart);"
       "stack[stacksize]=(vec2(rshort(xoff+istart),rshort(yoff+istart))+dx)/65536.*size;"
       "mx=max(mx,stack[0]);"
       "mn=min(mn,stack[0]);"
       "++stacksize;"
       "if(stacksize==2)"
         "d=min(d,lineseg(x,stack[0],stack[1]));"
       "else"
         " if(stacksize==3)"
           "d=min(d,spline2(stack[0],stack[1],stack[2],x));"
     "}"
   "glyphsize=abs(mx-mn);"
   "return d;"
 "}"
 "float dglyphpts(vec2 x,int ascii)"
 "{"
   "float nchars=rshort(0.),off=-1.;"
   "for(float i=0.;i<nchars;i+=1.)"
     "{"
       "int ord=int(rshort(1.+2.*i));"
       "if(ord==ascii)"
         "{"
           "off=rshort(1.+2.*i+1);"
           "break;"
         "}"
     "}"
   "if(off==-1.)"
     "return 1.;"
   "vec2 dx=mix(c.xx,c.zz,vec2(rshort(off),rshort(off+2.)))*vec2(rshort(off+1.),rshort(off+3.));"
   "float npts=rshort(off+4.),xoff=off+5.,yoff=off+6.+npts,d=1.;"
   "for(float i=0.;i<npts;i+=1.)"
     "{"
       "vec2 xa=(vec2(rshort(xoff+i),rshort(yoff+i))+dx)/65536.*size;"
       "d=min(d,length(x-xa)-.002);"
     "}"
   "return d;"
 "}"
 "mat2 rot(float t)"
 "{"
   "vec2 sc=vec2(cos(t),sin(t));"
   "return mat2(sc*c.xz,sc.yx);"
 "}"
 "float blend(float tstart,float tend,float dt)"
 "{"
   "return smoothstep(tstart-dt,tstart+dt,iTime)*(1.-smoothstep(tend-dt,tend+dt,iTime));"
 "}"
 "float softmin(float a,float b,float k)"
 "{"
   "float h=clamp(.5+.5*(b-a)/k,0.,1.);"
   "return mix(b,a,h)-k*h*(1.-h);"
 "}"
 "float softabs(float x,float a)"
 "{"
   "return-softmin(x,-x,a);"
 "}"
 "vec2 softabs(vec2 x,float a)"
 "{"
   "return-vec2(softmin(x.x,-x.x,a),softmin(x.y,-x.y,a));"
 "}"
 "float dtetrahedron(vec3 x,float a,float w)"
 "{"
   "return abs(softmin(lineseg(vec3(softabs(x.x,.5*a),x.yz),c.yyy,a*vec3(1.,0.,-1./sqrt(2.))),lineseg(vec3(x.x,softabs(x.y,.5*a),x.z),c.yyy,a*vec3(0.,1.,1./sqrt(2.))),.5*a))-w;"
 "}"
 "float dicosahedron(vec3 x,float a,float w)"
 "{"
   "mat3 r=rot(.3*sin(vec3(1.1,2.2,3.3)*iTime+.5*pi));"
   "float phi=.5*(1.+sqrt(5.)),d=softmin(spline2(c.yyy,.5*a*vec3(0.,1.,phi),a*r*vec3(0.,1.,phi),vec3(x.x,softabs(x.yz,.5*a))),spline2(c.yyy,.5*a*vec3(phi,0.,1.),a*r*vec3(phi,0.,1.),vec3(softabs(x.x,.5*a),x.y,softabs(x.z,.5*a))),.5*a);"
   "d=softmin(d,spline2(c.yyy,.5*a*vec3(1.,phi,0.),r*a*vec3(1.,phi,0.),vec3(softabs(x.xy,.5*a),x.z)),.5*a);"
   "return abs(d)-w;"
 "}"
 "vec2 talien(vec3 x,float a)"
 "{"
   "vec3 dt=.01*vec3(sin(iTime),cos(iTime),sin(iTime));"
   "float dr=.3*a;"
   "vec2 sdf=vec2(dicosahedron(x,.2,.04),1.);"
   "vec3 y=mod(x,dr)-.5*dr,ind=x-y;"
   "float da=dicosahedron(ind,.2,.04);"
   "if(abs(da)-.025<0.)"
     "sdf=add(sdf,vec2(length(y)-(.05+.1*rand(ind))*a,3.));"
   "float guard=-length(max(abs(y)-vec3(.5*dr*c.xx,.6),0.));"
   "guard=abs(guard)+dr*.1;"
   "sdf.x=min(sdf.x,guard);"
   "return sdf;"
 "}"
 "vec3 ind;"
 "vec2 scene1(vec3 x)"
 "{"
   "x+=2.*iTime*c.yxy-.05*x.y;"
   "float dr=.3;"
   "vec3 y=mod(x,dr)-.5*dr;"
   "float tlo=clamp(mfsnoise(x.xy,.1,.5,.4),-.1,.1),thi=mfsnoise(x.xy,.5,500.,.4),d=x.z+.2-.3*(.5*tlo+thi);"
   "d=min(d,x.z+1.-.1*thi);"
   "float guard=-length(max(abs(y)-vec3(.5*dr*c.xx,.6),0.));"
   "guard=abs(guard)+dr*.1;"
   "d=min(d,guard);"
   "vec2 sdf=vec2(d,1.),sda=vec2(x.z+1.,2.);"
   "sdf=mix(sdf,sda,step(sda.x,sdf.x));"
   "return sdf;"
 "}\n"
 "#define raymarch(scene,xc,ro,d,dir,s,N,eps,flag)flag=false;for(int i=0;i<N;++i){xc=ro+d*dir;s=scene(xc);if(s.x<eps){flag=true;break;}d+=s.x;}\n"
 "#define calcnormal(scene,n,eps,xc){float ss=scene(xc).x;n=normalize(vec3(scene(xc+eps*c.xyy).xc-ss,scene(xc+eps*c.yxy).xc-ss,scene(xc+eps*c.yyx).xc-ss));}\n"
 "#define camerasetup(camera,ro,r,u,t,uv,dir){camera(ro,r,u,t);t+=uv.x*r+uv.y*u;dir=normalize(t-ro);}\n"
 "#define post(color,uv){col=mix(clamp(col,c.yyy,c.xxx),c.xxx,smoothstep(1.5/iResolution.y,-1.5/iResolution.y,stroke(logo(uv-vec2(-.45,.45),.02),.005)));col+=vec3(0.,0.05,0.1)*sin(uv.y*1050.+5.*iTime);}\n"
 "void camera1(out vec3 ro,out vec3 r,out vec3 u,out vec3 t)"
 "{"
   "ro=.5*c.yyx,r=c.xyy,u=c.yyx+.3*c.yxy,t=c.yxy+.4*c.yyx;"
 "}"
 "vec3 synthcol(float scale,float phase)"
 "{"
   "vec3 c2=vec3(207.,30.,102.)/255.,c3=vec3(245.,194.,87.)/255.;"
   "mat3 r1=rot(.5*phase*vec3(1.1,1.3,1.5));"
   "return 1.1*mix(-cross(c2,r1*c2),-(r1*c2),scale);"
 "}"
 "vec3 stdcolor(vec2 x)"
 "{"
   "return.5+.5*cos(iTime+x.xyx+vec3(0,2,4));"
 "}"
 "bool hfloor=false;"
 "vec3 color(float rev,float ln,float index,vec2 uv,vec3 x)"
 "{"
   "vec3 col=c.yyy;"
   "if(index==1.)"
     "{"
       "x*=.01;"
       "vec3 c1=stdcolor(150.*x.z+x.xy+.5*rand(ind.xy+17.)),c2=stdcolor(150.*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)+11.+uv),c3=stdcolor(150.*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)+23.+uv);"
       "col=.1*c1*vec3(1.,1.,1.)+.2*c1*vec3(1.,1.,1.)*ln+1.5*vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.)))+2.*c1*pow(rev,8.)+3.*c1*pow(rev,16.);"
       "col=clamp(.23*col,0.,1.);"
     "}"
   "else"
     " if(index==2.)"
       "return x*=.1,hfloor=true,.5*stdcolor(x.xy+.5*rand(ind.xy+17.));"
     "else"
       " if(index==3.)"
         "{"
           "x*=.01;"
           "vec3 c1=stdcolor(150.*x.z+x.xy+.5*rand(ind.xy+27.)+iNBeats),c2=stdcolor(150.*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+12.)+iNBeats+21.+uv),c3=stdcolor(150.*x.z+x.xy+x.yz+x.zx+.5*rand(ind.xy+15.)+iNBeats+33.+uv);"
           "col=.4*c1*vec3(1.,1.,1.)+.2*c1*vec3(1.,1.,1.)*ln+.5*vec3(1.,1.,1.)*pow(rev,2.*(2.-1.5*clamp(iScale,0.,1.)))+2.*c1*pow(rev,8.)+3.*c1*pow(rev,16.);"
           "col=clamp(col,0.,1.);"
         "}"
   "return col;"
 "}"
 "vec4 thick(vec2 x,vec4 sdf,vec2 n)"
 "{"
   "for(int i=1;i<6;++i)"
     "sdf=add(vec4(stroke(sdf.x*n.x*n.y*2.*snoise((3.+4.*iScale)*x-2.-iTime-1.2),.01),.003/abs(sdf.x+.2*snoise(x-2.-iTime))*stdcolor(x+c.xx*.3*float(i))),sdf);"
   "return sdf;"
 "}"
 "vec4 geometry(vec2 x)"
 "{"
   "vec4 sdf=vec4(stroke(stroke(logo(x,.2),.06),.01),2.5*stdcolor(x*1.7));"
   "return sdf;"
 "}"
 "const float dx=.0001;"
 "vec2 normal(vec2 x)"
 "{"
   "float s=geometry(x).x;"
   "return normalize(vec2(geometry(x+dx*c.xy).x-s,geometry(x+dx*c.yx).x-s));"
 "}"
 "float star(vec2 x,float r0)"
 "{"
   "return 1.-smoothstep(.5*r0,r0,length(x));"
 "}"
 "vec3 bandc(vec2 x,float a)"
 "{"
   "return mix(c.yyy,c.xxy,step(.5*a,mod(x.x+x.y-.1*iTime,a)));"
 "}"
 "vec4 gir(vec2 x,float r)"
 "{"
   "vec4 sdf=vec4(dgear(x,vec2(r-.015,r),floor(107.143*r)),c.xxy);"
   "sdf=add(sdf,vec4(length(x)-.536*r,c.yyy));"
   "sdf=add(sdf,vec4(abs(length(x)-.321*r)-.036*r,c.xxy));"
   "return sdf;"
 "}"
 "vec3 background1(vec2 x)"
 "{"
   "float dr=.03,scale;"
   "vec2 y=mod(x,dr)-.5*dr;"
   "float rs=rand(x-y)*.005,dx=-.5*(dr-rs)+(dr-2.*rs)*rand(x-y+1.),dy=-.5*(dr-rs)+(dr-2.*rs)*rand(x-y+2.);"
   "scale=star(y-vec2(dx,dy),rs);"
   "vec3 color=scale*clamp(8.*rand(x.xy+4.)*stdcolor(rand(x-y+3.)*x.xy),0.,1.);"
   "float f=mfsnoise(x.xy-6.93,.2,100.,.55);"
   "color+=mix(c.yyy,stdcolor(x),.5+.95*f);"
   "color+=mix(c.yyy,2.*stdcolor(x+4.),.5+.33*f);"
   "color+=mix(c.yyy,stdcolor(x+8.),.5+.79*f);"
   "return clamp(color,0.,1.);"
 "}"
 "void mainImage(out vec4 fragColor,in vec2 fragCoord)"
 "{"
   "vec2 uv=fragCoord/iResolution.yy-.5;"
   "vec3 col=c.yyy;"
   "if(iTime<6.)"
     "{"
       "vec4 sdf=vec4(1.,col);"
       "float d=1.,dc=1.,dca=1.;"
       "vec2 vn=.01*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime),snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));"
       "{"
         "size=1.54;"
         "carriage=-.25*c.xy;"
         "int str[14]=int[14](72,101,108,108,111,32,85,78,67,32,50,48,49,56);"
         "for(int i=0;i<14;++i)"
           "{"
             "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
               "{"
                 "vec2 bound=uv-carriage-vn+.05*c.yx;"
                 "d=min(d,dglyph(bound,str[i]));"
                 "float d0=dglyphpts(bound,str[i]);"
                 "dc=min(dc,d0);"
                 "dca=min(dca,stroke(d0,.002));"
                 "carriage+=glyphsize.x*c.xy+.01*c.xy;"
               "}"
           "}"
       "}"
       "d=stroke(d,.0034)+.1*length(vn);"
       "sdf=add(sdf,vec4(d,c.xxx));"
       "sdf=add(sdf,vec4(dca,c.xxx));"
       "sdf=add(sdf,vec4(dc,c.xxy));"
       "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(1.,5.,1.);"
     "}"
   "else"
     " if(iTime<16.)"
       "{"
         "col=mix(clamp(col,c.yyy,c.xxx),bandc(uv,.1),smoothstep(1.5/iResolution.y,-1.5/iResolution.y,stroke(logo(uv-.5*c.xy,.2),.05)));"
         "col=mix(clamp(col,c.yyy,c.xxx),c.xxy,smoothstep(1.5/iResolution.y,-1.5/iResolution.y,stroke(stroke(logo(uv-.5*c.xy,.2),.05),.001)));"
         "float n=15.,dr=.112;"
         "mat2 r=rot(1.1*iTime),mr=rot(-1.1*iTime-2.*pi/n*dr);"
         "vec4 sdf=gir(r*(uv-.3*c.xy),dr);"
         "sdf=add(sdf,gir(mr*(uv-.09*c.xy),dr));"
         "sdf=add(sdf,gir(r*(uv-.7*c.xy),dr));"
         "sdf=add(sdf,gir(r*r*(uv+.065*c.xy),.5*dr));"
         "sdf=add(sdf,gir(mr*(uv+.22*c.xy),dr));"
         "col=mix(clamp(col,c.yyy,c.xxx),sdf.yzw,smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x))*blend(7.,15.,1.);"
       "}"
     "else"
       " if(iTime<22.)"
         "{"
           "vec4 sdf=vec4(1.,col);"
           "float d=1.,dc=1.,dca=1.;"
           "vec2 vn=.01*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime),snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));"
           "{"
             "size=1.54;"
             "carriage=-.25*c.xy;"
             "int str[17]=int[17](81,77,46,32,78,82,52,46,32,84,101,97,109,50,49,48,46);"
             "for(int i=0;i<17;++i)"
               "{"
                 "if(abs(uv.x)<1.5&&abs(uv.y)<.1)"
                   "{"
                     "vec2 bound=uv-carriage-vn+.05*c.yx;"
                     "d=min(d,dglyph(bound,str[i]));"
                     "float d0=dglyphpts(bound,str[i]);"
                     "dc=min(dc,d0);"
                     "dca=min(dca,stroke(d0,.002));"
                     "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                   "}"
               "}"
           "}"
           "d=stroke(d,.0034)+.1*length(vn);"
           "sdf=add(sdf,vec4(d,c.xxx));"
           "sdf=add(sdf,vec4(dca,c.xxx));"
           "sdf=add(sdf,vec4(dc,c.xxy));"
           "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(17.,21.,1.);"
         "}"
       "else"
         " if(iTime<42.)"
           "{"
             "vec3 ro,r,u,t,x,dir;"
             "camerasetup(camera1,ro,r,u,t,uv,dir);"
             "float d=(.15-ro.z)/dir.z;"
             "if(uv.y>.1)"
               "{"
                 "col=background1(uv);"
                 "post(col,uv);"
                 "fragColor=vec4(col*blend(23.,41.,1.),1.);"
                 "return;"
               "}"
             "else"
               "{"
                 "bool hit;"
                 "vec2 s;"
                 "raymarch(scene1,x,ro,d,dir,s,300,.0001,hit);"
                 "if(hit==false||x.y>12.)"
                   "{"
                     "col=background1(uv);"
                     "post(col,uv);"
                     "fragColor=vec4(col*blend(23.,41.,1.),1.);"
                     "return;"
                   "}"
                 "vec3 n;"
                 "calcnormal(scene1,n,.005,x);"
                 "vec3 l=x+2.*c.yyx,re=normalize(reflect(-l,n)),v=normalize(x-ro);"
                 "float rev=abs(dot(re,v)),ln=abs(dot(l,n));"
                 "col=color(rev,ln,s.y,uv,x);"
                 "if(s.y==2.)"
                   "{"
                     "for(float i=.7;i>.5;i-=.2)"
                       "{"
                         "dir=normalize(reflect(dir,n));"
                         "d=.05;"
                         "ro=x;"
                         "raymarch(scene1,x,ro,d,dir,s,50,.001,hit);"
                         "if(hit==false||x.y>12.)"
                           "{"
                             "col=mix(col,background1(uv),.5);"
                             "post(col,uv);"
                             "fragColor=vec4(col,1.);"
                             "break;"
                           "}"
                         "calcnormal(scene1,n,.001,x);"
                         "l=x+2.*c.yyx;"
                         "re=normalize(reflect(-l,n));"
                         "v=normalize(x-ro);"
                         "rev=abs(dot(re,v));"
                         "ln=abs(dot(l,n));"
                         "col=mix(col,color(rev,ln,s.y,uv,x),i);"
                       "}"
                   "}"
               "}"
             "col*=blend(23.,41.,1.);"
           "}"
         "else"
           " if(iTime<48.)"
             "{"
               "vec4 sdf=vec4(1.,col);"
               "float d=1.,dc=1.,dca=1.;"
               "vec2 vn=.01*vec2(snoise(1.36*uv-.66*vec2(1.5,2.4)*iTime),snoise(1.35*uv-.4*vec2(1.2,2.1)*iTime));"
               "{"
                 "size=1.04;"
                 "int str[24],nstr;"
                 "if(uv.y>.335+vn.y)"
                   "carriage=-.4*c.xy+.4*c.yx,str=int[24](68,101,107,97,100,101,110,99,101,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0);"
                 "else"
                   " if(uv.y>.235+vn.y)"
                     "carriage=-.4*c.xy+.3*c.yx,str=int[24](74,117,109,97,108,97,117,116,97,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0);"
                   "else"
                     " if(uv.y>.135+vn.y)"
                       "carriage=-.4*c.xy+.2*c.yx,str=int[24](75,101,119,108,101,114,115,32,38,32,77,70,88,46,0,0,0,0,0,0,0,0,0,0);"
                     "else"
                       " if(uv.y>.035+vn.y)"
                         "carriage=-.4*c.xy+.1*c.yx,str=int[24](68,97,109,111,110,101,115,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);"
                       "else"
                         " if(uv.y>-.065+vn.y)"
                           "carriage=-.4*c.xy,str=int[24](77,97,116,116,32,67,117,114,114,101,110,116,46,0,0,0,0,0,0,0,0,0,0,0);"
                         "else"
                           " if(uv.y>-.3+vn.y)"
                             "carriage=-.4*c.xy-.15*c.yx,str=int[24](83,99,104,110,97,40,112,112,115,124,114,99,104,41,103,105,114,108,115,46,0,0,0,0);"
                           "else"
                             "{"
                               "col=c.yyy;"
                               "post(col,uv);"
                               "fragColor=vec4(col,1.);"
                               "return;"
                             "}"
                 "for(int i=0;i<24;++i)"
                   "{"
                     "{"
                       "vec2 bound=uv-carriage-vn+.05*c.yx;"
                       "d=min(d,dglyph(bound,str[i]));"
                       "float d0=dglyphpts(bound,str[i]);"
                       "dc=min(dc,d0);"
                       "dca=min(dca,stroke(d0,.002));"
                       "carriage+=glyphsize.x*c.xy+.01*c.xy;"
                     "}"
                   "}"
               "}"
               "d=stroke(d,.0034)+.1*length(vn);"
               "sdf=add(sdf,vec4(d,c.xxx));"
               "sdf=add(sdf,vec4(dca,c.xxx));"
               "sdf=add(sdf,vec4(dc,c.xxy));"
               "col=sdf.yzw*smoothstep(1.5/iResolution.y,-1.5/iResolution.y,sdf.x)*blend(43.,47.,1.);"
             "}"
   "post(col,uv);"
   "fragColor=vec4(col,1.);"
 "}"
 "void main()"
 "{"
   "mainImage(gl_FragColor,gl_FragCoord.xy);"
 "}";

#endif // GFX_H_
