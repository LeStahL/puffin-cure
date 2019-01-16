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
 
#version 330

precision highp float;

uniform float iBlockOffset;
uniform float iSampleRate;
uniform float iVolume;

#define PI radians(180.)
float clip(float a) { return clamp(a,-1.,1.); }
float theta(float x) { return smoothstep(0., 0.01, x); }
float _sin(float a) { return sin(2. * PI * mod(a,1.)); }
float _sin(float a, float p) { return sin(2. * PI * mod(a,1.) + p); }
float _sq(float a) { return sign(2.*fract(a) - 1.); }
float _sq(float a,float pwm) { return sign(2.*fract(a) - 1. + pwm); }
float _psq(float a) { return clip(50.*_sin(a)); }
float _psq(float a, float pwm) { return clip(50.*(_sin(a) - pwm)); } 
float _tri(float a) { return (4.*abs(fract(a)-.5) - 1.); }
float _saw(float a) { return (2.*fract(a) - 1.); }
float quant(float a,float div,float invdiv) { return floor(div*a+.5)*invdiv; }
float freqC1(float note){ return 32.7 * pow(2.,note/12.); }
float minus1hochN(int n) { return (1. - 2.*float(n % 2)); }
float minus1hochNminus1halbe(int n) { return round(sin(.5*PI*float(n))); }
float pseudorandom(float x) { return fract(sin(dot(vec2(x),vec2(12.9898,78.233))) * 43758.5453); }

#define pat4(a,b,c,d,x) mod(x,1.)<.25 ? a : mod(x,1.)<.5 ? b : mod(x,1.) < .75 ? c : d

const float BPM = 63.;
const float BPS = BPM/60.;
const float SPB = 60./BPM;

const float Fsample = 44100.; // I think?
const float Tsample = 1./Fsample;

const float filterthreshold = 1e-3;

float doubleslope(float t, float a, float d, float s)
{
    return smoothstep(-.00001,a,t) - (1.-s) * smoothstep(0.,d,t-a);
}


float env_ADSR(float x, float L, float A, float D, float S, float R)
{
    float att = x/A;
    float dec = 1. - (1.-S)*(x-A)/D;
    float rel = (x <= L-R) ? 1. : (L-x)/R;
    return x<A ? att : x<A+D ? dec : x<= L-R ? S : x<=L ? (L-x)/R : 0.;
}


float s_atan(float a) { return 2./PI * atan(a); }
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


float comp_SAW(int N, float inv_N) {return inv_N * minus1hochN(N);}
float comp_TRI(int N, float inv_N) {return N % 2 == 0 ? 0. : inv_N * inv_N * minus1hochNminus1halbe(N);}
float comp_SQU(int N, float inv_N, float PW) {return N % 2 == 0 ? 0. : inv_N * (1. - minus1hochNminus1halbe(N))*_sin(PW);}
float comp_HAE(int N, float inv_N, float PW) {return N % 2 == 0 ? 0. : inv_N * (minus1hochN(N)*_sin(PW*float(N)+.25) - 1.);}

float MACESQ(float t, float f, float phase, int NMAX, int NINC, float MIX, float CO, float NDECAY, float RES, float RES_Q, float DET, float PW, int keyF)
{
    float ret = 0.;
    float INR = keyF==1 ? 1./CO : f/CO;
    float IRESQ = keyF==1 ? 1./RES_Q : 1./(RES_Q*f);
    
    float p = f*t + phase;
    for(int N=1; N<=NMAX; N+=NINC)
    {
        float float_N = float(N);
        float inv_N = 1./float_N;
        float comp_mix = MIX < 0. ? (MIX+1.) * comp_TRI(N,inv_N)    +  (-MIX)  * comp_SAW(N,inv_N)
                       : MIX < 1. ?   MIX    * comp_TRI(N,inv_N)    + (1.-MIX) * comp_SQU(N,inv_N,PW)
                                  : (MIX-1.) * comp_HAE(N,inv_N,PW) + (2.-MIX) * comp_SQU(N,inv_N,PW);

        float filter_N = pow(1. + pow(float_N*INR,NDECAY),-.5) + RES * exp(-pow((float_N*f-CO)*IRESQ,2.));
        
        if(abs(filter_N*comp_mix) < 1e-6) break; //or is it wise to break already?
        
        ret += comp_mix * filter_N * (_sin(float_N * p) + _sin(float_N * p * (1.+DET)));
    }
    return s_atan(ret);
}

float reverbFsaw3_IIR(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4)
{
    int imax = int(log(filterthreshold)/log(IIRgain));
    float delay[4] = float[4](IIRdel1, IIRdel2, IIRdel3, IIRdel4);
    
    float sum = 0.;
    
    // 4 IIR comb filters
    for(int d=0; d<8; d++)
    {
        float fac = 1.;
        
        for(int i=0; i<imax; i++)
        {
            float _TIME = time - float(i)*delay[d] * (.8 + .4*pseudorandom(sum));
            sum += fac*(theta(_TIME*SPB)*exp(-8.*_TIME*SPB)*((.5+(.5*_psq(8.*_TIME*SPB,0.)))*(2.*fract(f*_TIME+0.)-1.)));
            fac *= -IIRgain;
        }
    }
    return .25*sum;
}

float reverbFsaw3_AP1(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4, float APgain, float APdel1)
{
    // first allpass delay line
    float _TIME = time;
    float sum = -APgain * reverbFsaw3_IIR(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4);
    float fac = 1. - APgain * APgain;
    
    int imax = 1 + int((log(filterthreshold)-log(fac))/log(APgain));
    
    for(int i=0; i<imax; i++)
    {
        _TIME -= APdel1 * (.9 + 0.2*pseudorandom(time));
        sum += fac * reverbFsaw3_IIR(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4);
        fac *= APgain * (1. + 0.01*pseudorandom(_TIME));
    }
    return sum;        
}

float reverbFsaw3(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4, float APgain, float APdel1, float APdel2)
{   // // based on this Schroeder Reverb from Paul Wittschen: http://www.paulwittschen.com/files/schroeder_paper.pdf
    // todo: add some noise...
    // second allpass delay line
    float _TIME = time;
    float sum = -APgain * reverbFsaw3_AP1(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4, APgain, APdel1);
    float fac = 1. - APgain * APgain;

    int imax = 1 + int((log(filterthreshold)-log(fac))/log(APgain));

    for(int i=0; i<imax; i++)
    {
        _TIME -= APdel2 * (.9 + 0.2*pseudorandom(time));
        sum += fac * reverbFsaw3_AP1(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4, APgain, APdel1);
        fac *= APgain * (1. + 0.01*pseudorandom(_TIME));
    }
    return sum;        
}
float reverbsnrrev_IIR(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4)
{
    int imax = int(log(filterthreshold)/log(IIRgain));
    float delay[4] = float[4](IIRdel1, IIRdel2, IIRdel3, IIRdel4);
    
    float sum = 0.;
    
    // 4 IIR comb filters
    for(int d=0; d<8; d++)
    {
        float fac = 1.;
        
        for(int i=0; i<imax; i++)
        {
            float _TIME = time - float(i)*delay[d] * (.8 + .4*pseudorandom(sum));
            sum += fac*clamp(1.6*_tri(_TIME*(350.+(6000.-800.)*smoothstep(-.01,0.,-_TIME)+(800.-350.)*smoothstep(-.01-.01,-.01,-_TIME)))*smoothstep(-.1,-.01-.01,-_TIME) + .7*fract(sin(_TIME*90.)*4.5e4)*doubleslope(_TIME,.05,.3,.3),-1., 1.)*doubleslope(_TIME,0.,.25,.3);
            fac *= -IIRgain;
        }
    }
    return .25*sum;
}

float reverbsnrrev_AP1(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4, float APgain, float APdel1)
{
    // first allpass delay line
    float _TIME = time;
    float sum = -APgain * reverbsnrrev_IIR(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4);
    float fac = 1. - APgain * APgain;
    
    int imax = 1 + int((log(filterthreshold)-log(fac))/log(APgain));
    
    for(int i=0; i<imax; i++)
    {
        _TIME -= APdel1 * (.9 + 0.2*pseudorandom(time));
        sum += fac * reverbsnrrev_IIR(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4);
        fac *= APgain * (1. + 0.01*pseudorandom(_TIME));
    }
    return sum;        
}

float reverbsnrrev(float time, float f, float tL, float IIRgain, float IIRdel1, float IIRdel2, float IIRdel3, float IIRdel4, float APgain, float APdel1, float APdel2)
{   // // based on this Schroeder Reverb from Paul Wittschen: http://www.paulwittschen.com/files/schroeder_paper.pdf
    // todo: add some noise...
    // second allpass delay line
    float _TIME = time;
    float sum = -APgain * reverbsnrrev_AP1(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4, APgain, APdel1);
    float fac = 1. - APgain * APgain;

    int imax = 1 + int((log(filterthreshold)-log(fac))/log(APgain));

    for(int i=0; i<imax; i++)
    {
        _TIME -= APdel2 * (.9 + 0.2*pseudorandom(time));
        sum += fac * reverbsnrrev_AP1(_TIME, f, tL, IIRgain, IIRdel1, IIRdel2, IIRdel3, IIRdel4, APgain, APdel1);
        fac *= APgain * (1. + 0.01*pseudorandom(_TIME));
    }
    return sum;        
}




float AMAYSYN(float t, float B, float Bon, float Boff, float note, int Bsyn, float Brel)
{
    float Bprog = B-Bon;
    float Bproc = Bprog/(Boff-Bon);
    float L = Boff-Bon;
    float tL = SPB*L;
    float _t = SPB*(B-Bon);
    float f = freqC1(note);
    float vel = 1.; //implement later

    float env = theta(B-Bon) * (1. - smoothstep(Boff, Boff+Brel, B));
    float s = _sin(t*f);

    if(Bsyn == 0){}
    else if(Bsyn == 6){
      s = reverbFsaw3(_t,f,tL,.1,.00297,.00371,.00411,.00437,.3,.00017,.0005);env = theta(B-Bon)*pow(1.-smoothstep(Boff, Boff+Brel, B),.005);}
    
    else if(Bsyn == -2){
      s = 3.*s_atan(vel*smoothstep(0.,.015,_t)*smoothstep(.1+.15,.15,_t)*MACESQ(_t,(50.+(200.-50.)*smoothstep(-.12, 0.,-_t)),5.,10,1,.8,1.,1.,1.,.1,.1,0.,1) + .4*.5*step(_t,.03)*_sin(_t*1100.*1.*_saw(_t*800.*1.)) + .4*(1.-exp(-1000.*_t))*exp(-40.*_t)*_sin((400.-200.*_t)*_t*_sin(1.*(50.+(200.-50.)*smoothstep(-.12, 0.,-_t))*_t)));}
    else if(Bsyn == -6){
      s = .4*(.6+(.25*_psq(4.*B,0.)))*vel*fract(sin(t*100.*.3)*50000.*2.)*doubleslope(_t,0.,.05,0.);}
    else if(Bsyn == -7){
      s = vel*clamp(1.6*_tri(_t*(350.+(6000.-800.)*smoothstep(-.01,0.,-_t)+(800.-350.)*smoothstep(-.01-.01,-.01,-_t)))*smoothstep(-.1,-.01-.01,-_t) + .7*fract(sin(_t*90.)*4.5e4)*doubleslope(_t,.05,.3,.3),-1., 1.)*doubleslope(_t,0.,.25,.3);}
    else if(Bsyn == -8){
      s = reverbsnrrev(_t,f,tL,.15,.000297,.000371,.000411,.000437,.2,1.7e-05,5e-05);}
    
    return clamp(env,0.,1.) * s_atan(s);
}


float mainSynth(float time)
{
    int NO_trks = 3;
    int trk_sep[4] = int[4](0,7,15,17);
    int trk_syn[3] = int[3](6,23,23);
    float trk_norm[3] = float[3](.7,1.,1.);
    float trk_rel[3] = float[3](1.3,1.,1.);
    float mod_on[17] = float[17](4.,8.,12.,16.,20.,24.,28.,16.,18.,20.,22.,24.,26.,28.,30.,0.,2.);
    float mod_off[17] = float[17](8.,12.,16.,20.,24.,28.,32.,18.,20.,22.,24.,26.,28.,30.,32.,2.,4.);
    int mod_ptn[17] = int[17](0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2);
    float mod_transp[17] = float[17](0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.);
    float max_mod_off = 34.;
    int drum_index = 23;
    float drum_synths = 9.;
    int NO_ptns = 3;
    int ptn_sep[4] = int[4](0,32,36,40);
    float note_on[40] = float[40](0.,.125,.25,.375,.5,.625,.75,.875,1.,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.,2.125,2.25,2.375,2.5,2.625,2.75,2.875,3.,3.125,3.25,3.375,3.5,3.625,3.75,3.875,0.,.5,1.,1.5,0.,.5,1.25,1.75);
    float note_off[40] = float[40](.125,.25,.375,.5,.625,.75,.875,1.,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.,2.125,2.25,2.375,2.5,2.625,2.75,2.875,3.,3.125,3.25,3.375,3.5,3.625,3.75,3.875,4.,.5,1.,1.5,2.,.125,1.25,2.,2.);
    float note_pitch[40] = float[40](17.,22.,24.,29.,31.,36.,38.,43.,48.,46.,41.,36.,33.,29.,44.,39.,36.,32.,41.,36.,32.,29.,26.,22.,41.,38.,34.,32.,37.,34.,31.,27.,2.,2.,2.,2.,6.,8.,7.,8.);
    float note_vel[40] = float[40](1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.);
    
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

        int _modU = TLEN-1;
        for(int i=0; i<TLEN-1; i++) if(BT < mod_on[(trk_sep[trk]+i)]) {_modU = i; break;}
               
        int _modL = TLEN-1;
        for(int i=0; i<TLEN-1; i++) if(BT < mod_off[(trk_sep[trk]+i)] + trk_rel[trk]) {_modL = i; break;}
       
        for(int _mod = _modL; _mod <= _modU; _mod++)
        {
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

                if(trk_syn[trk] == drum_index)
                {
                    int Bdrum = int(mod(note_pitch[ptn_sep[ptn]+_note], drum_synths));
                    float Bvel = note_vel[(ptn_sep[ptn]+_note)] * pow(2.,mod_transp[trk_sep[trk]+_mod]/6.);

                    //0 is for sidechaining - am I doing this right?
                    if(Bdrum == 0)
                        r_sidechain = (1. - theta(B-Bon) * exp(-1000.*(B-Bon))) * smoothstep(Bon,Boff,B);
                    else
                        d += trk_norm[trk] * AMAYSYN(time, B, Bon, Boff, Bvel, -Bdrum, trk_rel[trk]);
                }
                else
                {
                    r += trk_norm[trk] * AMAYSYN(time, B, Bon, Boff,
                                                   note_pitch[(ptn_sep[ptn]+_note)] + mod_transp[trk_sep[trk]+_mod], trk_syn[trk], trk_rel[trk]);
                }
            }
        }
    }

    return s_atan(s_atan(r_sidechain * r + d));
}

vec2 mainSound(float t)
{
    //enhance the stereo feel
//     return vec2(sin(2.*PI*440.*t));
//     float stereo_delay = 2e-4;
    return vec2(-1.+2.*mod(t,1.));
//     return vec2(mainSynth(t), mainSynth(t-stereo_delay));
}

void main() 
{
    // Rescale fragment coordinates to match the correct memory
    vec2 x = gl_FragCoord.xy;
    
    // compute time `t` based on the pixel we're about to write
    // the 512.0 means the texture is 512 pixels across so it's
    // using a 2 dimensional texture, 512 samples per row
    float t = iBlockOffset + dot(x, vec2(1.,512.0))/iSampleRate;
    
    // Get the 2 values for left and right channels
    vec2 y = mainSound( t );

    // convert them from -1 to 1 to 0 to 65535
    vec2 v  = floor((.5+.5*clamp(y,-1.,1.))*65535.);

    // separate them into low and high bytes
    vec2 vl = floor(mod(v,256.));
    vec2 vh = floor(v/256.);

    // write them out where 
    // RED   = channel 0 low byte
    // GREEN = channel 0 high byte
    // BLUE  = channel 1 low byte
    // ALPHA = channel 2 high byte
    gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y)/255.;
}

