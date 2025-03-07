#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

#define STIM //if df simulate the stimulated IF model
// #define EIF //if def simulate exponential IF model else leaky IF model

// simulation parameters
const int num_real = 200000; // number of realizations
const int steps = 1e6; // total number of integration steps, only even numbers
const double dt = 1e-4; // integration step 
const double T = steps * dt; // total integrated time
const double f_nyquist = 0.5 / dt; // maximal frequency that can be sampled with dt steps 
const double v_init = 0; // initial voltage

// physical parameters
const double v_T = 1; // threshold voltage
const double v_R = 0; // reset voltage
const double mu = 0.8; // mean input 
const double Delta_v = 1; // parameter of EIF
const double v_t = 0.8; // soft threshold value of the EIF
const double D = 0.1; // noise intensity 
const double f_stim_max = 100; // max frequency of broadband stimulus
const double s2 = 1; // stimulus variance
const double eps = 0.31622777; // stimulus amplitude = sqrt(0.1)
// const double eps = 1; // stimulus amplitude 
const int tau_ref = 5000; // refractory period in units of simulation steps

/*
 * set real (double) Fourier trafo input array
 */
void set_double_array(double* a, double* b, const int len)
{
    for(int i = 0; i < len; i++)
    {
        b[i] = a[i];
    }
}

/*
 * set complex (fftw_complex) Fourier trafo output array and fix time spacing 
 */
void set_complex_array(fftw_complex* a, fftw_complex* b, const int len)
{
    for(int l = 0; l < len; l++)
    {
        b[l] = a[l] * dt;
    }
}

/*
 * create Gaussian random numbers using the Marsaglia polar method
 */
void get_gauss(double* grnd)
{
    double u1, u2, r1, r2, y1, y2, w;
		do{

            // create two uniform random numbers in [0,1]
            u1 = (double) rand() / (double) RAND_MAX;
            u2 = (double) rand() / (double) RAND_MAX;

            // create two uniform random numbers in [-1,1]
			r1 = 2.0 * u1 - 1.0;
			r2 = 2.0 * u2 - 1.0;
			w = r1 * r1 + r2 * r2;
		}while(w >= 1.0 || w == 0);

	w = sqrt((-2.0 * log(w) ) / w);
	y1 = r1 * w;
	y2 = r2 * w;
	grnd[0] = y1;
	grnd[1] = y2;
}

/*
 * create seed of the random number generator based on current time
 */
int gus() {
	struct timeval tp;
	gettimeofday(&tp,0);
	return tp.tv_usec;
}

/*
 * calculate f(v) voltage behaviour in the IF model can be non-linear, for LIF f(v) = -v + mu, for EIF f(v) = -v + mu + delta_v * exp((v-v_T)/delta_v)
 */ 
#if defined(EIF)
    double get_v_func(double v)
    {
        return -v + mu + Delta_v * exp((v - v_t) / Delta_v);
    }
#else
    double get_v_func(double v)
    {
        return -v + mu;
    }
#endif

/*
 * calculate voltage change
 */ 
double get_dv(double f, double noise)
{
    return f * dt + noise * dt;
} 

/*
 * calculate voltage change with external stimulus
 */ 
double get_dv_stim(double f, double noise, double stim)
{
    return f * dt + noise * dt + eps * stim * dt;
} 

/*
 * calculate white noise
 */ 
double get_white_noise(double xrnd)
{
    return sqrt(2 * D / dt) * xrnd;
}

/*
 * calculate stochastic broadband stimulus for all times by creating the FT of the stimulus based on its power spectrum
 * for |f| > f_stim_max the spectrum vanishes and is constant (=S) for |f| <= f_stim_max where f_stim_max <= f_nyquist = 2 / dt 
 */
void get_stim_stoch(double* stims, const int N_frq, const size_t frqBytes)
{
    fftw_complex *stim_ft;
    fftw_plan p;
    double rnd[2];
    double frq;
    
    // set constant value of stimulus power spectrum based on stimulus variance
    const double S = s2 / (2 * f_stim_max);
    
    // setup real and imaginary part of the stimulus Fourier transform
    stim_ft = (fftw_complex*) fftw_malloc(frqBytes);
    memset(stim_ft, 0, frqBytes);

    // set real und imaginary part based on power spectrum
    for(int l = 0; l < N_frq; l++)
    {
        frq = l / T;
        if(frq <= f_stim_max)
        {
        get_gauss(rnd);
        stim_ft[l] = sqrt(S * T / 2)* rnd[0] + I * sqrt(S * T / 2) * rnd[1];
        }
    }

    // perform inverse Fourier trafo (complex to real) to obtain stimulus time series
    p = fftw_plan_dft_c2r_1d(steps, stim_ft, stims, FFTW_ESTIMATE); // make plan, ie, choose a FFT algo
    fftw_execute(p); // execute plan, ie, calc the inverse Fourier trafo
    
    // free the input array and delete the plan
    fftw_destroy_plan(p);
    fftw_free(stim_ft);

    // multiply with df = 1/T to fix the frequency spacing which was assumed to be 1 
    for(int i = 0; i < steps; i++)
    {
        stims[i] = stims[i] * 1 / T;
    }
}

/*
 * calculate cross/power spectra of a single realization, ie  c = ab* / T 
 */
void calc_spec(fftw_complex* a, fftw_complex* b, fftw_complex* c, const int N_frq)
{
    for(int l = 0; l < N_frq; l++)
    {
        c[l] = a[l] * conj(b[l]);
        c[l] = c[l] / T;
    }
}

/*
 * calculating the new average in a realization-wise calculation of element-wise array average
 */
void reali_avg(fftw_complex* a, fftw_complex* b, const int reali, const int N_frq)
{
    for(int l = 0; l < N_frq; l++)
    {
        a[l] = reali * a[l] + b[l];
        a[l] = a[l] / (reali + 1);
    }
}

/*
 * handle memory alloc, averaging and spectrum calculation
 */
void calc_spectra(fftw_complex* a_ft, fftw_complex* b_ft, fftw_complex* S_ab, const int N_frq, const int reali_idx)
{
    size_t frqBytes = N_frq * sizeof(fftw_complex);
    fftw_complex *s_ab;
    s_ab = (fftw_complex*) fftw_malloc(frqBytes);
    memset(s_ab, 0, frqBytes);

    calc_spec(a_ft, b_ft, s_ab, N_frq);

    // add current realization to the averaging
    reali_avg(S_ab, s_ab, reali_idx, N_frq);
    
    fftw_free(s_ab);
}

/*
 * calculate the susceptibilty, ie X(w) = S_xs(w) / S_ss(w)
 */
void calc_sus(fftw_complex* S_xs, fftw_complex* S_ss, fftw_complex* sus, const int N_frq)
{
    double frq;
    for(int l = 0; l < N_frq; l++)
    {
        frq = l / T;
        if(frq <= f_stim_max)
        {
            sus[l] = S_xs[l] / (eps * S_ss[l]);
        }
    }
}

void save_timeseries()
{
    double grnd[2];
    int rndid;
    double vrnd;
    
    double dv; // voltage increment
    size_t stepBytes = steps * sizeof(double); // size of data arrays in time domain

    // setup voltage, spike-train and noise
    double v, vs, x, xs, noise, f;

    // counting the timesteps during the refactory period
    int timer, timers;

    // determine  number of frequency bin in Fourier space, ie in range [0, f_nyquist]
    int N_frq;
        if(steps % 2 == 0)
            N_frq = steps / 2 + 1;
        else
        {
            printf("Number of integration steps must be even!\n");
            exit(1);
        }
    size_t frqBytes = N_frq * sizeof(fftw_complex);

    double stim; // external stimulus at given time

    // setup stochastic stimulus at all times
    double *stims;
    stims = (double *)malloc(stepBytes);

    // calculate stochastic broadband stimulus for all times
    get_stim_stoch(stims, N_frq, frqBytes);
    stim = stims[0]; // set initial stimulus

    // create two gaussian random numbers
    get_gauss(grnd);

    // setup spiketrain, initial voltage and initial noise
    x = 0;
    xs = 0;
    v = v_init;
    vs = v_init;

    timer = tau_ref;
    timers = tau_ref;

    noise = get_white_noise(grnd[0]);

    FILE *timeseriesFile;
    #if defined(EIF)
        if(tau_ref == 0)
            {
                timeseriesFile = fopen("./data/ts_EIF_white.txt", "w");
            }
            else
            {
                timeseriesFile = fopen("./data/ts_EIF_ref.txt", "w");
            }
    #else
        if(tau_ref == 0)
        {
            timeseriesFile = fopen("./data/ts_LIF_white.txt", "w");
        }
        else
        {
            timeseriesFile = fopen("./data/ts_LIF_ref.txt", "w");
        }
    #endif

    fprintf(timeseriesFile, "%f %f %f %f %f %f %f\n", 0, v, vs, x, xs, noise, stim);
    // integrate Langevin equation
    for (int i = 1; i < steps; i++)
    {

        if (timers < tau_ref)
        {
            dv = 0;
            timers++;
        }
        else
        {
            f = get_v_func(vs);
            dv = get_dv_stim(f, noise, stim);
        }
        
        vs = vs + dv;

        if (timer < tau_ref)
        {
            dv = 0;
            timer++;
        }
        else
        {
        f = get_v_func(v);
        dv = get_dv(f, noise);
        }

        v = v + dv;

        // check for threshold and reset if necessary
        if (v >= v_T)
        {
            v = v_R;
            x = 1 / dt; // discrete delta peak
            timer = 0;
        }
        if (vs >= v_T)
        {
            vs = v_R;
            xs = 1 / dt; // discrete delta peak
            timers = 0;
        }

        // update noise
        rndid = i % 2;
        vrnd = grnd[rndid];

        noise = get_white_noise(vrnd);

        // create new random numbers if necessary
        if (rndid == 1)
        {
            get_gauss(grnd);
        }
        
        // update stimulus
        stim = stims[i];

        // save timeseries
        fprintf(timeseriesFile, "%f %f %f %f %f %f %f\n", i*dt, v, vs, x, xs, noise, stim);

        // reset spike trains
        x = 0;
        xs = 0;
        }    
    fclose(timeseriesFile);
    free(stims);
}

int main()
{
    // seed random number generator
    unsigned long long seed;
    seed=gus();
    srand(seed);
    double grnd[2];
    int rndid;
    double vrnd;

    // runtime info
    time_t start, end;
    time(&start);
    int simulation_time;
    FILE *infoFile;

    double dv; // voltage increment
    size_t stepBytes = steps * sizeof(double); // size of data arrays in time domain

    // setup voltage, spike-train, noise and model-specific function array
    double *v, *x;
    v = (double *)malloc(stepBytes);  
    x = (double *)malloc(stepBytes);
    double noise;


    #if defined(EIF)
        double f;
        double *fs;
        fs = (double *)malloc(stepBytes);
    #else
        double f;
    #endif

    int count = 0; //spike counter

    int timer = tau_ref;

    // setup real to complex Fourier transformation
    // determine  number of frequency bin in Fourier space, ie in range [0, f_nyquist]
    int N_frq;
        if(steps % 2 == 0)
            N_frq = steps / 2 + 1;
        else
        {
            printf("Number of integration steps must be even!\n");
            exit(1);
        }
    size_t frqBytes = N_frq * sizeof(fftw_complex);

    // setup Fourier trafo plan
    double *in; // real input
    fftw_complex *out; // complex output
    fftw_plan p;

    in = (double*)malloc(stepBytes);
    out = (fftw_complex*) fftw_malloc(frqBytes);
    memset(in, 0, stepBytes);
    memset(out, 0, frqBytes);

    p = fftw_plan_dft_r2c_1d(steps, in, out, FFTW_MEASURE);

    // stimulus setup
    #if defined(STIM)
        double stim; // external stimulus at given time

        // setup stochastic stimulus at all times
        double *stims;
        stims = (double *)malloc(stepBytes);

        // setup spike-train-stimulus cross spectrum
        fftw_complex *S_xs;
        S_xs = (fftw_complex*) fftw_malloc(frqBytes);
        memset(S_xs, 0, frqBytes);

        // setup voltage-stimulus cross spectrum
        fftw_complex *S_vs;
        S_vs = (fftw_complex*) fftw_malloc(frqBytes);
        memset(S_vs, 0, frqBytes);

        // setup stimulus power spectrum
        fftw_complex *S_ss;
        S_ss = (fftw_complex*) fftw_malloc(frqBytes);
        memset(S_ss, 0, frqBytes);

        #if defined(EIF)
            // setup v_func-stimulus cross spectrum
            fftw_complex *S_fs;
            S_fs = (fftw_complex*) fftw_malloc(frqBytes);
            memset(S_fs, 0, frqBytes);
        #endif

        // save voltage, spike and stimulus timeseries of a single realisation
        save_timeseries();
    #else
        // setup spectra of spontaneous activity
        fftw_complex *S_xx, *S_xv, *S_vv;
        S_xx = (fftw_complex*) fftw_malloc(frqBytes);
        S_xv = (fftw_complex*) fftw_malloc(frqBytes);
        S_vv = (fftw_complex*) fftw_malloc(frqBytes);
        memset(S_xx, 0, frqBytes);
        memset(S_xv, 0, frqBytes);
        memset(S_vv, 0, frqBytes);

        #if defined(EIF)
            fftw_complex *S_xf, *S_vf;
            S_xf = (fftw_complex*) fftw_malloc(frqBytes);
            S_vf = (fftw_complex*) fftw_malloc(frqBytes);
            memset(S_xf, 0, frqBytes);
            memset(S_vf, 0, frqBytes);
        #endif
    #endif

    // set initial voltage and noise
    v[steps-1] = v_init;

    // create two gaussian random numbers
    get_gauss(grnd);
    noise = get_white_noise(grnd[0]);

    // for all realizations
    for (int k = 0; k < num_real; k++)
    {
        #if defined(STIM)
            // calculate stochastic broadband stimulus for all times
            get_stim_stoch(stims, N_frq, frqBytes);
            stim = stims[0]; // set initial stimulus
        #endif

        // setup spiketrain and continue voltage and OUP noise dynamics of previous realization
        memset(x, 0, stepBytes);
        v[0] = v[steps-1];

        // setup model-specific function
        f = get_v_func(v[0]);
        #if defined(EIF)
            fs[0] = f;
        #endif

        // integrate Langevin equation
        for (int i = 1; i < steps; i++)
        {            
            if (timer < tau_ref)
            {
                dv = 0;
                timer++;
            }
            else
            {
                #if defined(STIM)
                    dv = get_dv_stim(f, noise, stim);
                #else
                    dv = get_dv(f, noise);
                #endif
            }

            v[i] = v[i-1] + dv;

            // check for threshold and reset if necessary
            if (v[i] >= v_T)
            {
                v[i] = v_R;
                x[i] = 1 / dt; // discrete delta peak
                count++;
                timer = 0;
            }

            // update stimulus
            #if defined(STIM)
                stim = stims[i];
            #endif

            // update noise
            rndid = i % 2;
            vrnd = grnd[rndid];

            noise = get_white_noise(vrnd);

            // update model-specific function
            f = get_v_func(v[i]);
            #if defined(EIF)
                fs[i] = f;
            #endif

            // create new random numbers if necessary
            if (rndid == 1)
            {
                get_gauss(grnd);
            }
        }

        // spectra calculation
        #if defined(STIM)
            // stimulus power spectra and stimulus-spike-train cross spectra
            fftw_complex *x_ft, *v_ft, *s_ft;
            x_ft = (fftw_complex*) fftw_malloc(frqBytes);
            v_ft = (fftw_complex*) fftw_malloc(frqBytes);
            s_ft = (fftw_complex*) fftw_malloc(frqBytes);
            memset(x_ft, 0, frqBytes);
            memset(v_ft, 0, frqBytes);
            memset(s_ft, 0, frqBytes);

            set_double_array(x, in, steps);
            fftw_execute(p);
            set_complex_array(out, x_ft, N_frq);
            
            set_double_array(v, in, steps);
            fftw_execute(p);
            set_complex_array(out, v_ft, N_frq);

            set_double_array(stims, in, steps);
            fftw_execute(p);
            set_complex_array(out, s_ft, N_frq);

            calc_spectra(x_ft, s_ft, S_xs, N_frq, k);
            calc_spectra(v_ft, s_ft, S_vs, N_frq, k);
            calc_spectra(s_ft, s_ft, S_ss, N_frq, k);

            fftw_free(x_ft);  
            fftw_free(v_ft);  

            #if defined(EIF)
                fftw_complex *fs_ft;
                fs_ft = (fftw_complex*) fftw_malloc(frqBytes);
                memset(fs_ft, 0, frqBytes);

                set_double_array(fs, in, steps);
                fftw_execute(p);
                set_complex_array(out, fs_ft, N_frq);

                calc_spectra(fs_ft, s_ft, S_fs, N_frq, k);

                fftw_free(fs_ft);
            #endif

            fftw_free(s_ft);
        #else
            // spike-train power spectra, spike-train-voltage cross spectrum and noise power spectrum
            fftw_complex *v_ft, *x_ft;
            v_ft = (fftw_complex*) fftw_malloc(frqBytes);
            x_ft = (fftw_complex*) fftw_malloc(frqBytes);
            memset(v_ft, 0, frqBytes);
            memset(x_ft, 0, frqBytes);
           
            set_double_array(v, in, steps);
            fftw_execute(p);
            set_complex_array(out, v_ft, N_frq);
            
            set_double_array(x, in, steps);
            fftw_execute(p);
            set_complex_array(out, x_ft, N_frq);

            calc_spectra(x_ft, x_ft, S_xx, N_frq, k);
            calc_spectra(x_ft, v_ft, S_xv, N_frq, k);
            calc_spectra(v_ft, v_ft, S_vv, N_frq, k);

            #if defined(EIF)
                fftw_complex *fs_ft;
                fs_ft = (fftw_complex*) fftw_malloc(frqBytes);
                memset(fs_ft, 0, frqBytes);

                set_double_array(fs, in, steps);
                fftw_execute(p);
                set_complex_array(out, fs_ft, N_frq);

                calc_spectra(x_ft, fs_ft, S_xf, N_frq, k);
                calc_spectra(v_ft, fs_ft, S_vf, N_frq, k);

                fftw_free(fs_ft);
            #endif
            fftw_free(v_ft);
            fftw_free(x_ft);  
        #endif
    }

    free(x);
    free(v);
    free(in);
    fftw_free(out);
    fftw_destroy_plan(p);

    #if defined(EIF)
        free(fs);
    #endif

    // calulate firing rate 
    double rate;
    rate = count / (T * num_real);

    #if defined(STIM)
        free(stims);

        // calculate susceptibilities
        fftw_complex *sus_x, *sus_v;
        sus_x = (fftw_complex*) fftw_malloc(frqBytes);
        sus_v = (fftw_complex*) fftw_malloc(frqBytes);
        memset(sus_x, 0, frqBytes);
        memset(sus_v, 0, frqBytes);
        calc_sus(S_xs, S_ss, sus_x, N_frq);
        calc_sus(S_vs, S_ss, sus_v, N_frq);

        #if defined(EIF)
            fftw_complex *sus_f;
            sus_f = (fftw_complex*) fftw_malloc(frqBytes);
            memset(sus_f, 0, frqBytes);
            calc_sus(S_fs, S_ss, sus_f, N_frq);
        #endif 
    #endif
    
    // write spectra to file
    FILE *resultsFile;

    #if defined(STIM)
        #if defined(EIF)
            if(tau_ref == 0)
            {
                resultsFile = fopen("./data/res_s_EIF_white.txt", "w");
            }
            else
            {
                resultsFile = fopen("./data/res_s_EIF_ref.txt", "w");
            }
        #else
            if(tau_ref == 0)
            {
                resultsFile = fopen("./data/res_s_LIF_white.txt", "w");
            }
            else
            {
                resultsFile = fopen("./data/res_s_LIF_ref.txt", "w");
            }
        #endif

        for(int l = 0; l < N_frq; l++)
        {
            fprintf(resultsFile, "%f %f %f %f %f %f %f %f %f %f %f", l / T, creal(S_xs[l]), cimag(S_xs[l]), creal(S_vs[l]), cimag(S_vs[l]), creal(S_ss[l]), cimag(S_ss[l]), creal(sus_x[l]), cimag(sus_x[l]), creal(sus_v[l]), cimag(sus_v[l]));
            #if defined(EIF)
                fprintf(resultsFile, " %f %f", creal(sus_f[l]), cimag(sus_f[l]));
            #endif
            fprintf(resultsFile, "\n");
            }
    #else
        #if defined(EIF)
            if(tau_ref == 0)
            {
                resultsFile = fopen("./data/res_EIF_white.txt", "w");
            }
            else
            {
                resultsFile = fopen("./data/res_EIF_ref.txt", "w");
            }
        #else
            if(tau_ref == 0)
            {
                resultsFile = fopen("./data/res_LIF_white.txt", "w");
            }
            else
            {
                resultsFile = fopen("./data/res_LIF_ref.txt", "w");
            }
        #endif
        for(int l = 0; l < N_frq; l++)
            { 
                fprintf(resultsFile, "%f %f %f %f %f %f %f", l / T, creal(S_xx[l]), cimag(S_xx[l]), creal(S_xv[l]), cimag(S_xv[l]), creal(S_vv[l]), cimag(S_vv[l]));
                #if defined(EIF)
                    fprintf(resultsFile, " %f %f %f %f", creal(S_xf[l]), cimag(S_xf[l]), creal(S_vf[l]), cimag(S_vf[l]));
                #endif
                fprintf(resultsFile, "\n");
            }
    #endif
    fclose(resultsFile);

    #if defined(STIM)
        fftw_free(S_xs);
        fftw_free(S_vs);
        fftw_free(S_ss);
        fftw_free(sus_x);
        fftw_free(sus_v);
        #if defined(EIF)
            fftw_free(S_fs);
            fftw_free(sus_f);
        #endif
    #else
        fftw_free(S_xx);
        fftw_free(S_xv);
        fftw_free(S_vv);
        #if defined(EIF)
            fftw_free(S_xf);
            fftw_free(S_vf);
        #endif
    #endif

    // calculate simulation and save all parameters
    #if defined(STIM)
        #if defined(EIF)
            if(tau_ref == 0)
            {
                infoFile = fopen("./data/info/EIF_white_s.txt", "w");
            }
            else
            {
                infoFile = fopen("./data/info/EIF_ref_s.txt", "w");
            }
        #else
            if(tau_ref == 0)
            {
                infoFile = fopen("./data/info/LIF_white_s.txt", "w");
            }
            else
            {
                infoFile = fopen("./data/info/LIF_ref_s.txt", "w");
            }
        #endif
    #else
        #if defined(EIF)
            if(tau_ref == 0)
            {
                infoFile = fopen("./data/info/EIF_white.txt", "w");
            }
            else
            {
                infoFile = fopen("./data/info/EIF_ref.txt", "w");
            }
        #else
            if(tau_ref == 0)
            {
                infoFile = fopen("./data/info/LIF_white.txt", "w");
            }
            else
            {
                infoFile = fopen("./data/info/LIF_ref.txt", "w");
            }
        #endif        
    #endif
    time(&end);
    simulation_time = (int)difftime(end, start);
    fprintf(infoFile, "Simulation runtime= %d minutes %d sec\n", (int)(simulation_time/60),(simulation_time%60));
    fprintf(infoFile, "Number of realizations = %d\n", num_real);
    fprintf(infoFile, "Total number of integration steps = %d\n", steps);
    fprintf(infoFile, "Time increment = %f\n", dt);
    fprintf(infoFile, "Nyquist frequency = %.1f\n", 0.5 / dt);
    fprintf(infoFile, "Total simlated time = %f\n", steps * dt);
    fprintf(infoFile, "Initial voltage = %.2f\n", v_init);
    fprintf(infoFile, "Threshold voltage = %.2f\n", v_T);
    fprintf(infoFile, "Reset voltage = %.2f\n", v_R);
    fprintf(infoFile, "Mean input = %.2f\n", mu);
    fprintf(infoFile, "Noise intensity = %.2f\n", D);
    fprintf(infoFile, "Refactory period = %f\n", tau_ref * dt);
    fprintf(infoFile, "Average firing rate = %f", rate);
    #if defined(STIM)
        fprintf(infoFile, "\nStochastic broadband stimulus over [%.1f, %.1f]\n",-f_stim_max, f_stim_max);
        fprintf(infoFile, "Stimulus variance = %.2f\n", s2);
        fprintf(infoFile, "Stimulus amplitude = %f", eps);
    #endif
    #if defined(EIF)
        fprintf(resultsFile, "\nEIF parameter, Delta_v = %.2f", Delta_v);
    #endif

    fclose(infoFile);

    return 0;
}