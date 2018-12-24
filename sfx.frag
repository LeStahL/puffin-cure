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

#define PI radians(180.)
float clip(float a) { return clamp(a,-1.,1.); }
float theta(float x) { return smoothstep(0., 0.01, x); }
float _sin(float a) { return sin(2. * PI * mod(a,1.)); }
float _sin(float a, float p) { return sin(2. * PI * mod(a,1.) + p); }
float _unisin(float a,float b) { return (.5*_sin(a) + .5*_sin((1.+b)*a)); }
float _sq(float a) { return sign(2.*fract(a) - 1.); }
float _sq(float a,float pwm) { return sign(2.*fract(a) - 1. + pwm); }
float _psq(float a) { return clip(50.*_sin(a)); }
float _psq(float a, float pwm) { return clip(50.*(_sin(a) - pwm)); } 
float _tri(float a) { return (4.*abs(fract(a)-.5) - 1.); }
float _saw(float a) { return (2.*fract(a) - 1.); }
float quant(float a,float div,float invdiv) { return floor(div*a+.5)*invdiv; }
float quanti(float a,float div) { return floor(div*a+.5)/div; }
float freqC1(float note){ return 32.7 * pow(2.,note/12.); }
float minus1hochN(int n) { return (1. - 2.*float(n % 2)); }

#define pat4(a,b,c,d,x) mod(x,1.)<.25 ? a : mod(x,1.)<.5 ? b : mod(x,1.) < .75 ? c : d

const float BPM = 80.;
const float BPS = BPM/60.;
const float SPB = 60./BPM;

const float Fsample = 44100.; // I think?

float doubleslope(float t, float a, float d, float s)
{
    return smoothstep(-.00001,a,t) - (1.-s) * smoothstep(0.,d,t-a);
}

float s_atan(float a) { return 2./PI * atan(a); }
float s_crzy(float amp) { return clamp( s_atan(amp) - 0.1*cos(0.9*amp*exp(amp)), -1., 1.); }
float squarey(float a, float edge) { return abs(a) < edge ? a : floor(4.*a+.5)*.25; } 

float supershape(float s, float amt, float A, float B, float C, float D, float E)
{
    float w;
    float m = sign(s);
    s = abs(s);

    if(s<A) w = B * smoothstep(0.,A,s);
    else if(s<C) w = C + (B-C) * smoothstep(C,A,s);
    else if(s<=D) w = s;
    else if(s<=1.)
    {
        float _s = (s-D)/(1.-D);
        w = D + (E-D) * (1.5*_s*(1.-.33*_s*_s));
    }
    else return 1.;
    
    return m*mix(s,w,amt);
}

float GAC(float t, float offset, float a, float b, float c, float d, float e, float f, float g)
{
    t = t - offset;
    return t<0. ? 0. : a + b*t + c*t*t + d*sin(e*t) + f*exp(-g*t);
}

float MACESQ(float t, float f, int MAXN, float MIX, float INR, float NDECAY, float RES, float RES_Q, float DET, float PW)
{
    float ret = 0.;
    
    int Ninc = 8; // try this: leaving out harmonics...
    
    float p = f*t;
    for(int N=0; N<=MAXN; N+=Ninc)
    {
        float mode     = 2.*float(N) + 1.;
        float inv_mode = 1./mode; 		// avoid division? save table of Nmax <= 20 in some array or whatever
        float comp_TRI = (N % 2 == 1 ? -1. : 1.) * inv_mode*inv_mode;
        float comp_SQU = inv_mode * (1. + (2.*float(N%2)-1.)*_sin(PW)); 
        float comp_mix = (MIX * comp_TRI + (1.-MIX) * comp_SQU);
        
        //one special mode from legacy code 'matzeskuh' - I computed some shitty-but-fun Fourier coefficients for PWM
        if(MIX < -.01) comp_mix = 1./(2.*PI*float(N)) * (minus1hochN(N)*_sin(PW*float(N)+.25) - 1.);

        float filter_N = pow(1. + pow(float(N) * INR,2.*NDECAY),-.5) + RES * exp(-pow(float(N)*INR*RES_Q,2.));

        if(abs(filter_N*comp_mix) < 1e-6) break;
        
        ret += comp_mix * filter_N * (_sin(mode * p) + _sin(mode * p * (1.+DET)));
    }
    return s_atan(ret);
}

float QMACESQ(float t, float f, float QUANT, int MAXN, float MIX, float INR, float NDECAY, float RES, float RES_Q, float DET, float PW)
{
    return MACESQ(quant(t,QUANT,1./QUANT), f, MAXN, MIX, INR, NDECAY, RES, RES_Q, DET, PW);
}

float env_ADSR(float x, float L, float A, float D, float S, float R)
{
    float att = x/A;
    float dec = 1. - (1.-S)*(x-A)/D;
    float rel = (x <= L-R) ? 1. : (L-x)/R;
    return (x < A ? att : (x < A+D ? dec : S)) * rel;    
}

// CUDOS TO metabog https://www.shadertoy.com/view/XljSD3 - thanks for letting me steal
float resolpsomesaw1(float time, float f, float fa, float reso)
{
    int maxTaps = 128;
    fa= sqrt(fa);
    float c = pow(0.5, (128.0-fa*128.0)   / 16.0);
  	float r = pow(0.5, (reso*128.0+24.0) / 16.0);
    
    float v0 = 0.;
    float v1 = 0.;
    
    for(int i = 0; i < maxTaps; i++)
    {
          float _TIME = time - float(maxTaps-i)*(1.0/44100.0);
          float inp = (2.*fract(f*_TIME+0.)-1.);
          v0 =  (1.0-r*c)*v0  -  (c)*v1  + (c)*inp;
  		  v1 =  (1.0-r*c)*v1  +  (c)*v0;
    }
    
    return v1;
}


float env_ADSRexp(float x, float L, float A, float D, float S, float R)
{
    float att = pow(x/A,8.);
    float dec = S + (1.-S) * exp(-(x-A)/D);
    float rel = (x <= L-R) ? 1. : pow((L-x)/R,4.);
    return (x < A ? att : dec) * rel;    
}

//matze: not happy that I include this as is, but we'll just live with it, it is ancient knowledge
float bitexplosion(float time, float B, int dmaxN, float fvar, float B2amt, float var1, float var2, float var3, float decvar)
{
    float snd = 0.;
    float B2 = mod(B,2.);
    float f = 60.*fvar;
	float dt = var1 * 2.*PI/15. * B/sqrt(10.*var2-.5*var3*B);
    int maxN = 10 + dmaxN;
    for(int i=0; i<2*maxN+1; i++)
    {
        float t = time + float(i - maxN)*dt;
        snd += _sin(f*t + .5*(1.+B2amt*B2)*_sin(.5*f*t));
    }
    float env = exp(-2.*decvar*B);
    return atan(snd * env);
}

float AMAYSYN(float t, float B, float Bon, float Boff, float note, int Bsyn)
{
    float Bprog = B-Bon;
    float Bproc = Bprog/(Boff-Bon);
    float L = Boff-Bon;
    float tL = SPB*L;
    float _t = SPB*(B-Bon);
    float f = freqC1(note);
	float vel = 1.;

    float env = theta(B-Bon) * theta(Boff-B);
	float s = _sin(t*f);

	if(Bsyn == 0){}
    else if(Bsyn == 1){
      s = ((s_atan(_sin(.5*f*(t-0.0))+_sin((1.-.01)*.5*f*(t-0.0)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-0.0),_sin(.5*f*(t-0.0)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-0.0)-0.0))
      +_sin(2.5198*.5*f*((t-0.0)-1.0e-02))
      +_sin(2.5198*.5*f*((t-0.0)-2.0e-02))
      +_sin(2.5198*.5*f*((t-0.0)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-3.0e-01))+_sin((1.-.01)*.5*f*(t-3.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-3.0e-01),_sin(.5*f*(t-3.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-3.0e-01)-0.0))
      +_sin(2.5198*.5*f*((t-3.0e-01)-1.0e-02))
      +_sin(2.5198*.5*f*((t-3.0e-01)-2.0e-02))
      +_sin(2.5198*.5*f*((t-3.0e-01)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-6.0e-01))+_sin((1.-.01)*.5*f*(t-6.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-6.0e-01),_sin(.5*f*(t-6.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-6.0e-01)-0.0))
      +_sin(2.5198*.5*f*((t-6.0e-01)-1.0e-02))
      +_sin(2.5198*.5*f*((t-6.0e-01)-2.0e-02))
      +_sin(2.5198*.5*f*((t-6.0e-01)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-9.0e-01))+_sin((1.-.01)*.5*f*(t-9.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-9.0e-01),_sin(.5*f*(t-9.0e-01)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-9.0e-01)-0.0))
      +_sin(2.5198*.5*f*((t-9.0e-01)-1.0e-02))
      +_sin(2.5198*.5*f*((t-9.0e-01)-2.0e-02))
      +_sin(2.5198*.5*f*((t-9.0e-01)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-1.2))+_sin((1.-.01)*.5*f*(t-1.2)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-1.2),_sin(.5*f*(t-1.2)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-1.2)-0.0))
      +_sin(2.5198*.5*f*((t-1.2)-1.0e-02))
      +_sin(2.5198*.5*f*((t-1.2)-2.0e-02))
      +_sin(2.5198*.5*f*((t-1.2)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-1.5))+_sin((1.-.01)*.5*f*(t-1.5)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-1.5),_sin(.5*f*(t-1.5)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-1.5)-0.0))
      +_sin(2.5198*.5*f*((t-1.5)-1.0e-02))
      +_sin(2.5198*.5*f*((t-1.5)-2.0e-02))
      +_sin(2.5198*.5*f*((t-1.5)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-1.8))+_sin((1.-.01)*.5*f*(t-1.8)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-1.8),_sin(.5*f*(t-1.8)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-1.8)-0.0))
      +_sin(2.5198*.5*f*((t-1.8)-1.0e-02))
      +_sin(2.5198*.5*f*((t-1.8)-2.0e-02))
      +_sin(2.5198*.5*f*((t-1.8)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-2.1))+_sin((1.-.01)*.5*f*(t-2.1)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-2.1),_sin(.5*f*(t-2.1)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-2.1)-0.0))
      +_sin(2.5198*.5*f*((t-2.1)-1.0e-02))
      +_sin(2.5198*.5*f*((t-2.1)-2.0e-02))
      +_sin(2.5198*.5*f*((t-2.1)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-2.4))+_sin((1.-.01)*.5*f*(t-2.4)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-2.4),_sin(.5*f*(t-2.4)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-2.4)-0.0))
      +_sin(2.5198*.5*f*((t-2.4)-1.0e-02))
      +_sin(2.5198*.5*f*((t-2.4)-2.0e-02))
      +_sin(2.5198*.5*f*((t-2.4)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.)))
      +(s_atan(_sin(.5*f*(t-2.7))+_sin((1.-.01)*.5*f*(t-2.7)))+theta(Bprog)*exp(-.03*Bprog)*.15*_sin(.5*f*(t-2.7),_sin(.5*f*(t-2.7)))+theta(Bprog)*exp(-.03*Bprog)*.8*(_sin(2.5198*.5*f*((t-2.7)-0.0))
      +_sin(2.5198*.5*f*((t-2.7)-1.0e-02))
      +_sin(2.5198*.5*f*((t-2.7)-2.0e-02))
      +_sin(2.5198*.5*f*((t-2.7)-3.0e-02)))*(.7+.5*_tri(.5*Bprog+0.))));}
    else if(Bsyn == 2){
      s = _sin(.5*f*t)*env_ADSR(_t,tL,.5,2.,.5,.1);}
    else if(Bsyn == 3){
      s = theta(Bprog)*exp(-16.*mod(Bprog,.125))*theta(Bprog)*exp(-1.5*Bprog)*(s_atan((2.*fract(f*t+0.)-1.)+(2.*fract((1.-.01)*f*t+0.)-1.)+(2.*fract((1.-.033)*f*t+0.)-1.)+(2.*fract((1.-.04)*f*t+0.)-1.))+.6*s_atan((2.*fract(.5*f*t+.01)-1.)+(2.*fract((1.-.05)*.5*f*t+.01)-1.)+(2.*fract((1.-0.03)*.5*f*t+.01)-1.)+(2.*fract((1.-0.02)*.5*f*t+.01)-1.)));}
    else if(Bsyn == 4){
      s = _sin(f*t)
      +0.1*GAC(t,0.,1.,2.,0.5,3.,2.,2.,0.25)*_sin(f*t)
      +.1*GAC(t,0.,1.,2.,0.5,3.,2.,2.,0.25)*supershape(_sin(f*t),1.,.01,.7,.1,.6,.8);}
    else if(Bsyn == 5){
      s = resolpsomesaw1(t,f,.5*fract(B),.2);}
    else if(Bsyn == -1){
      s = s_atan(vel*(smoothstep(0.,.1,_t)*smoothstep(-(.1+.3),-.1,-_t)*(clip(10.*_tri((81.3+(208.8-81.3)*smoothstep(-.3, 0.,-_t))*t))+_sin(.5*(81.3+(208.8-81.3)*smoothstep(-.3, 0.,-_t))*t)))+ 1.2*step(_t,.05)*_sin(5000.*t*.8*_saw(1000.*t*.8)));}
    else if(Bsyn == -2){
      s = vel*fract(sin(t*100.*.9)*50000.*.9)*doubleslope(t,.03,.15,.15);}
    else if(Bsyn == -3){
      s = vel*bitexplosion(t, Bprog, 1,1.,1.,1.,1.,1.,1.);}
    
	return clamp(env,0.,1.) * s_atan(s);
}

float BA8(float x, int pattern)
{
    x = mod(x,1.);
    float ret = 0.;
	for(int b = 0; b < 8; b++)
    	if ((pattern & (1<<b)) > 0) ret += step(x,float(7-b)/8.);
    return ret * .125;
}

float mainSynth(float time)
{
    int NO_trks = 1;
    int trk_sep[2] = int[2](0,1);
    int trk_syn[1] = int[1](5);
    float trk_norm[1] = float[1](.9);
    float trk_rel[1] = float[1](.7);
    float mod_on[1] = float[1](0.);
    float mod_off[1] = float[1](8.);
    int mod_ptn[1] = int[1](0);
    float mod_transp[1] = float[1](0.);
    float max_mod_off = 8.;
    int drum_index = 6;
    float drum_synths = 1.;
    int NO_ptns = 1;
    int ptn_sep[2] = int[2](0,17);
    float note_on[17] = float[17](0.,1.,1.5,2.,2.5,3.,4.,4.,4.5,4.5,5.,5.5,5.5,6.,6.5,7.,7.5);
    float note_off[17] = float[17](1.,1.5,2.,2.5,3.,4.,4.5,4.5,5.,5.,5.5,6.,6.,7.,7.,8.,8.);
    float note_pitch[17] = float[17](31.,35.,33.,36.,33.,27.,50.,35.,33.,49.,36.,39.,45.,28.,40.,24.,39.);
    float note_vel[17] = float[17](1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.);
    
   
    float r = 0.;
    float d = 0.;

    // mod for looping
    float BT = mod(BPS * time, max_mod_off);
    if(BT > max_mod_off) return r;
    time = SPB * BT;

    float r_sidechain = 1.;

    float Bon = 0.;
    float Boff = 0.;

    for(int trk = 0; trk < NO_trks; trk++)
    {
        int TLEN = trk_sep[trk+1] - trk_sep[trk];
       
        int _mod = TLEN;
        for(int i=0; i<TLEN; i++) if(BT < mod_off[(trk_sep[trk]+i)]) {_mod = i; break;}
        if(_mod == TLEN) continue;
       
        float B = BT - mod_on[trk_sep[trk]+_mod];

        int ptn = mod_ptn[trk_sep[trk]+_mod];
        int PLEN = ptn_sep[ptn+1] - ptn_sep[ptn];
       
        int _noteU = PLEN-1;
        for(int i=0; i<PLEN-1; i++) if(B < note_on[(ptn_sep[ptn]+i+1)]) {_noteU = i; break;}

        int _noteL = PLEN-1;
        for(int i=0; i<PLEN-1; i++) if(B <= note_off[(ptn_sep[ptn]+i)] + trk_rel[trk]) {_noteL = i; break;}
       
        for(int _note = _noteL; _note <= _noteU; _note++)
        {
            Bon    = note_on[(ptn_sep[ptn]+_note)];
            Boff   = note_off[(ptn_sep[ptn]+_note)];

            float anticlick = 1.-exp(-1000.*(B-Bon)); //multiply this elsewhere?

            if(trk_syn[trk] == drum_index)
            {
                int Bdrum = int(mod(note_pitch[ptn_sep[ptn]+_note], drum_synths));
                float Bvel = note_vel[(ptn_sep[ptn]+_note)] * pow(2.,mod_transp[_mod]/6.);

                float d = 0.;

                if(Bdrum == 0) // Sidechain - have to multiply, actually?
                {
                    r_sidechain = anticlick - .999 * theta(B-Bon) * smoothstep(Boff,Bon,B);
                }
                else d += trk_norm[trk] * AMAYSYN(time, B, Bon, Boff, Bvel, -Bdrum);
            }
            else
            {
                r += trk_norm[trk] * AMAYSYN(time, B, Bon, Boff,
                                               note_pitch[(ptn_sep[ptn]+_note)] + mod_transp[_mod], trk_syn[trk]);
            }

        }
    }

    return s_atan(s_atan(r_sidechain * r + d));
//    return sign(snd) * sqrt(abs(snd)); // eine von Matzes "besseren" Ideen
}

vec2 mainSound(float t)
{
    //maybe this works in enhancing the stereo feel
    float stereo_width = 0.1;
    float stereo_delay = 0.00001;
   
    //float comp_l = mainSynth(t) + stereo_width * mainSynth(t - stereo_delay);
    //float comp_r = mainSynth(t) + stereo_width * mainSynth(t + stereo_delay);
   
    //return vec2(comp_l * .99999, comp_r * .99999);
   
    return vec2(mainSynth(t));
}


void main() 
{
   // compute time `t` based on the pixel we're about to write
   // the 512.0 means the texture is 512 pixels across so it's
   // using a 2 dimensional texture, 512 samples per row
   float t = iBlockOffset + ((gl_FragCoord.x-0.5) + (gl_FragCoord.y-0.5)*512.0)/iSampleRate;
    
//    t = mod(t, 4.5);
    
   // Get the 2 values for left and right channels
   vec2 y = iVolume * mainSound( t );

   // convert them from -1 to 1 to 0 to 65536
   vec2 v  = floor((0.5+0.5*y)*65536.0);

   // separate them into low and high bytes
   vec2 vl = mod(v,256.0)/255.0;
   vec2 vh = floor(v/256.0)/255.0;

   // write them out where 
   // RED   = channel 0 low byte
   // GREEN = channel 0 high byte
   // BLUE  = channel 1 low byte
   // ALPHA = channel 2 high byte
   gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
}

