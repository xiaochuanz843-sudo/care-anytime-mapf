#include <cstdlib>
#include <string>
#include <cstdio>   // ANCHORFIX_PORT: fprintf for layer WARN/FATAL diagnostics
static int g_amor_best = 1000000000; // AMOR_RR_PORT record-to-record best-global tracker
#include "LNS.h"
#include "ECBS.h"
#include <queue>
#include <random>
#include <algorithm>
#include <limits>
#include <map>
#include <cmath>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/random_device.hpp>

// ---- AMOR_SPSA_PORT: file-static learned-acceptance-policy state (one TACKLE run per process, like g_amor_best) ----
// theta = [bias, stagnation_w, time_w, GAP_w]; acting delta_eff = softplus(theta . [1, s, u, g]).
static double g_sp_th[4]    = {0.0, 0.0, 0.0, 0.0};
static int    g_sp_delta[4] = {1, 1, 1, 1};   // current Rademacher +/-1 SPSA perturbation (one draw per +/- pair)
static int    g_sp_phase    = 0;              // 0 = acting theta + c*Delta (measure R+), 1 = acting theta - c*Delta (measure R-)
static double g_sp_Rplus    = 0.0;            // reward from the phase-0 window
static long   g_sp_wc       = 0;              // window iteration counter
static int    g_sp_wstart   = 1000000000;     // best sum_of_costs at window start
static long   g_sp_isi      = 0;              // iterations since last global-best improvement
static std::vector<Path> g_sp_best_paths;
static std::vector<Path> g_bestret_paths;     // BESTRET_PORT: UNIVERSAL best-incumbent snapshot (ALL accept arms), restored at end of run()     // best-incumbent path snapshot (spsa best-return)

// ---- TK_REPAIR_PORT: file-static repair-order-bandit state (safe-arm-set delayed replan-priority bandit) ----
// eps-greedy-EMA over 5 fixed PP replan-priority rules (TK_REPAIR=20; unset/0 = stock random_shuffle).
// Arms: 0=random(stock) 1=longest-haul 2=shortest 3=most-delayed 4=least-delayed.
// DEFAULT arm set "0,2,3,4" EXCLUDES arm 1 (catastrophic on our fork); override with TK_RB_ARMS.
static double g_rb2_val[5] = {0,0,0,0,0};   // per-arm EMA of the slack-fraction reward
static int    g_rb2_n[5]   = {0,0,0,0,0};   // per-arm pull counts
static int    g_rb2_last   = -1;            // arm pulled for the in-flight neighborhood (-1 = none)
static double g_rb2_lb     = 0.0;           // free-flow SoC lower bound of the in-flight neighborhood
static double g_rb2_eps    = 0.15;          // TK_RB_EPS
static double g_rb2_alpha  = 0.2;           // TK_RB_ALPHA
static bool   g_rb2_allow[5] = {true,false,true,true,true};   // default TK_RB_ARMS="0,2,3,4" (safe arm set)
static bool   g_rb2_cfged  = false;
static void rb2_cfg()
{
    if (g_rb2_cfged) return;
    g_rb2_cfged = true; const char* s;
    if ((s = std::getenv("TK_RB_EPS")))   g_rb2_eps   = atof(s);
    if ((s = std::getenv("TK_RB_ALPHA"))) g_rb2_alpha = atof(s);
    if ((s = std::getenv("TK_RB_ARMS"))) { for (int a = 0; a < 5; a++) g_rb2_allow[a] = false;
        std::string v = s; size_t p = 0;
        while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size();
            int a = atoi(v.substr(p, q - p).c_str()); if (a >= 0 && a < 5) g_rb2_allow[a] = true; p = q + 1; }
        bool _any = false; for (int a = 0; a < 5; a++) if (g_rb2_allow[a]) _any = true;   // ANCHORFIX_PORT (fix D): empty arm set => UB in explore branch
        if (!_any) { g_rb2_allow[0] = g_rb2_allow[2] = g_rb2_allow[3] = g_rb2_allow[4] = true; g_rb2_allow[1] = false;
            fprintf(stderr, "[layer] WARN empty RB_ARMS, fallback to 0,2,3,4\n"); } }
}

// ---- CART_PORT: self-tuning budget-decayed record-to-record (TK_ACCEPT=cart) ----
static double g_cart_vol=0.0; static int g_cart_prev=-1; static const double g_cart_volbeta=1.0/32.0;
// ---- SATA_PORT: Simulated-Annealing / Threshold-Accepting accept modes (see add_saTA.py) ----
static std::mt19937 g_sata_rng(20240607u);                 // SATA_PORT: SEPARATE RNG (stock stays bit-identical when mode off)
static inline double g_sata_u01(){ return (double)g_sata_rng() / 4294967296.0; } // SATA_PORT: uniform [0,1)
// ---- REPV5_PORT: Beta-Thompson-Sampling repair-order bandit (TK_REPAIR=v5) with rawdelta Bernoulli reward ----
// Shares the 5 arms + allow-set (g_rb2_allow / rb2_cfg) with the eps-greedy bandit; only selection
// (Thompson sampling over Beta posteriors) and reward (magnitude-aware Bernoulli) differ.
static double g_ts_a[5] = {1,1,1,1,1};      // Beta alpha per arm (uniform Beta(1,1) prior)
static double g_ts_b[5] = {1,1,1,1,1};      // Beta beta  per arm
static double g_ts_scale = 16.0;            // TK_TS_SCALE: raw SoC-delta -> success-prob scale
static int    g_ts_cfged = 0;
static void ts_cfg(){ if (g_ts_cfged) return; g_ts_cfged = 1; const char* s;
    if ((s = std::getenv("TK_TS_SCALE"))) g_ts_scale = atof(s); }
static inline double ts_u01(){ return ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0); }
static inline double ts_normal(){ double u1 = ts_u01(), u2 = ts_u01();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2); }
static double ts_gamma(double k){ // Marsaglia-Tsang standard gamma, shape k >= 1 (alpha,beta always >= 1 here)
    double d = k - 1.0/3.0, c = 1.0/std::sqrt(9.0*d);
    for(;;){ double x = ts_normal(), v = 1.0 + c*x; if (v <= 0.0) continue; v = v*v*v; double u = ts_u01();
        if (u < 1.0 - 0.0331*x*x*x*x) return d*v;
        if (std::log(u) < 0.5*x*x + d*(1.0 - v + std::log(v))) return d*v; } }
static inline double ts_beta(double a, double b){ double x = ts_gamma(a), y = ts_gamma(b); return x/(x+y); }

// ---- TRAJ_PORT: anytime incumbent-trajectory dump (TK_TRAJ=<path>) ----
static std::ofstream g_traj_os; static long g_traj_prev = 2147483647L; static int g_traj_open = 0;
LNS::LNS(const Instance& instance, double time_limit, const string & init_algo_name, const string & replan_algo_name,
         const string & destroy_name, int neighbor_size, int num_of_iterations, bool use_init_lns,
         const string & init_destroy_name, bool use_sipp, int screen, PIBTPPS_option pipp_option, 
         const string & bandit_algorithm_name, int neighborhoodSizes,
         int num_agent,
         string algorithm, double epsilon, double decay, double k, int regions, string b)  :
         BasicLNS(instance, time_limit, neighbor_size, screen, bandit_algorithm_name, neighborhoodSizes, DESTROY_COUNT),
         init_algo_name(init_algo_name),  replan_algo_name(replan_algo_name), num_of_iterations(num_of_iterations),
         use_init_lns(use_init_lns),init_destroy_name(init_destroy_name),
         path_table(instance.map_size), pipp_option(pipp_option), num_agent(num_agent)
{
    q_values = new std::vector<int>(num_agent, INT_MAX-1);
    frequency = new std::vector<int>(num_agent, 0);
    location_frequency = new std::vector<int>();
    location_q_values = new std::vector<double>();
    start_time = Time::now();
    replan_time_limit = time_limit / 100;
    // Handel different ALNS options:
    if (destroy_name == "Adaptive")
    {
        if (b == "canonical"){
            alns_bernoulie = BCANONICAL;
            algorithm = "canonical";
        }
        else if (b == "add"){
            alns_bernoulie = ADD;
        }
        else if (b == "replace"){
            alns_bernoulie = REPLACE;
        }
        int numHeuristics = b == "add" ? DESTROY_COUNT+1 : DESTROY_COUNT;
        ALNS = true;
        heuristicBanditStats.destroy_weights.assign(numHeuristics, 1);
        heuristicBanditStats.destroy_weights_squared.assign(numHeuristics, 1);
        heuristicBanditStats.destroy_counts.assign(numHeuristics, 0);
        decay_factor = 0;
        reaction_factor = 0;
        if(numberOfNeighborhoodSizeCandidates > 0)
        {
            for(int index = 0; index < numHeuristics; index++)
            {
                neighborhoodBanditStats.push_back(new BanditStats());
                neighborhoodBanditStats[index]->destroy_weights.assign(numberOfNeighborhoodSizeCandidates, 1);
                neighborhoodBanditStats[index]->destroy_weights_squared.assign(numberOfNeighborhoodSizeCandidates, 1);
                neighborhoodBanditStats[index]->destroy_counts.assign(numberOfNeighborhoodSizeCandidates, 0);
            }
        }
    }
    else if (destroy_name == "RandomWalk")
        destroy_strategy = RANDOMWALK;
    else if (destroy_name == "Intersection")
        destroy_strategy = INTERSECTION;
    else if (destroy_name == "Random")
        destroy_strategy = RANDOMAGENTS;
    else
    {
        cerr << "Destroy heuristic " << destroy_name << " does not exists. " << endl;
        exit(-1);
    }
     this->epsilon = epsilon;
    this->decay = decay;
    this->k = k;
    // Handel different bandit implementations:
    if (algorithm == "canonical"){
        algo = CANONICAL;
    }
    else if (algorithm == "greedy"){
        this->epsilon = 0;
        algo = GREEDY;
    } else if (algorithm == "epsilon"){
        algo = EPSILON;
    } else if (algorithm == "decay"){
        algo = EPSILON_DECAY;
    } else if (algorithm == "ucb"){
        algo = UCB;
    } else if (algorithm == "topk-greedy"){
        this->epsilon = 0;
        algo = TOPK_EPSILON;
    } else if (algorithm == "topk-epsilon"){
        algo = TOPK_EPSILON;
    } else if (algorithm == "topk-decay"){
        algo = TOP_EPSILON_DECAY;
    } else if (algorithm == "topk-ucb"){
        algo = TOPK_UCB;
    } else if (algorithm == "bernoulie"){
        algo = BERNOULIE;
    } else if (algorithm == "random"){
        algo = RANDOM_SEED;
    } else if (algorithm == "roulette"){
        algo = ROULETTE_SEED;
    } else if (algorithm == "normal"){
        algo = NORMAL;
    } else if (algorithm == "tackle-tabu"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::TABU_INTENT_BASED;
        nonStationaryBandits = true;
    } else if (algorithm == "tackle-roulette"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::ROULETTE_INTENT_BASED;
        nonStationaryBandits = true;
    } else if (algorithm == "tackle-greedy"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::GREEDY_INTENT_BASED;
        nonStationaryBandits = true;
    } else if (algorithm == "tackle-random"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::RANDOM_INTENT_BASED;
        nonStationaryBandits = true;
    } else if (algorithm == "tackle-constant"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::CONSTANT_INTENT_BASED;
        nonStationaryBandits = true;
    } else if (algorithm == "tackle-tabu-stationary"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::TABU_INTENT_BASED;
        nonStationaryBandits = false;
    } else if (algorithm == "tackle-roulette-stationary"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::ROULETTE_INTENT_BASED;
        nonStationaryBandits = false;
    } else if (algorithm == "tackle-greedy-stationary"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::GREEDY_INTENT_BASED;
        nonStationaryBandits = false;
    } else if (algorithm == "tackle-random-stationary"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::RANDOM_INTENT_BASED;
        nonStationaryBandits = false;
    } else if (algorithm == "tackle-constant-stationary"){
        algo = TACKLE_SMALL;
        intentStrategy = intent_strategy::CONSTANT_INTENT_BASED;
        nonStationaryBandits = false;
    }

    if(algo == TACKLE_SMALL)
    {
        int upperBound = 1;
        if(nonStationaryBandits) {
            upperBound += k;
        } else {
            upperBound += num_agent;
        }
        for(int i = 0; i < upperBound; i++)
        {
            counterfactual_alpha.push_back(std::vector<int>(k, 1));
            counterfactual_beta.push_back(std::vector<int>(k, 1));
        }
    }
    
    this->regions = regions;

    this->neighbor_size = 8;
    alpha = std::vector<int>(num_agent, 1);
    beta = std::vector<int>(num_agent, 1);
    location_alpha = std::vector<int>(regions, 1);
    location_beta = std::vector<int>(regions, 1);
    mu = std::vector<double>(num_agent, 0.0);
    sigma2 = std::vector<double>(num_agent, 1.0);
    location_mu = std::vector<double>();
    location_sigma2 = std::vector<double>();
    int N = instance.getDefaultNumberOfAgents();
    agents.reserve(N);
    for (int i = 0; i < N; i++)
        agents.emplace_back(instance, i, use_sipp);
    preprocessing_time = ((fsec)(Time::now() - start_time)).count();
    if (screen >= 2)
        cout << "Pre-processing time = " << preprocessing_time << " seconds." << endl;
}

bool LNS::run()
{
    static const int    _tk_spsa   = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="spsa")?1:0; }(); // AMOR_SPSA_PORT
    static const int    _tk_spsa_W = [](){ const char* s=std::getenv("TK_SPSA_W"); return s?std::max(1,atoi(s)):64; }();
    static const double _tk_spsa_A = [](){ const char* s=std::getenv("TK_SPSA_A"); return s?atof(s):0.05; }();
    static const double _tk_gap_w  = [](){ const char* s=std::getenv("TK_GAP_W");  return s?atof(s):0.0; }();
    { const char* _tp = std::getenv("TK_TRAJ");   // TRAJ_PORT: per-run open, header once
        if (_tp && _tp[0] && !g_traj_open) { g_traj_os.open(_tp); g_traj_open = g_traj_os.is_open() ? 1 : 0; if (!g_traj_open) fprintf(stderr, "[layer] WARN: cannot open TRAJ path %s\n", _tp); g_traj_prev = 2147483647L;
            if (g_traj_os.is_open()) g_traj_os << "runtime,incumbent\n"; } }
    static const int _tk_cart = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="cart")?1:0; }(); // CART_PORT
    static const bool _amor_on = [](){ const char* a=std::getenv("TK_ACCEPT"); const char* r=std::getenv("TK_REPAIR"); const char* t=std::getenv("TK_TRAJ");
        return (a&&a[0])||(r&&r[0])||(t&&t[0]); }(); // ANCHORFIX_PORT: layer master switch (non-empty check, consistent with L2/BL/AD)
    if (_amor_on && replan_algo_name != "PP") { fprintf(stderr, "[layer] FATAL: accept/repair layer requires PP replan (replan_algo_name=%s)\n", replan_algo_name.c_str()); exit(86); } // ANCHORFIX_PORT fix E (fail-fast; EECBS/CBS as INIT is fine)
    { const char* _ea = std::getenv("TK_ACCEPT"); const char* _er = std::getenv("TK_REPAIR"); // ANCHORFIX_PORT fix G: unknown env values must not silently degrade to stock
      if (_ea && _ea[0] && std::string(_ea)!="rr" && std::string(_ea)!="cart" && std::string(_ea)!="spsa" && std::string(_ea)!="sa" && std::string(_ea)!="ta")
          { fprintf(stderr, "[layer] FATAL: unknown TK_ACCEPT=%s (want rr|cart|spsa|sa|ta)\n", _ea); exit(87); }
      if (_er && _er[0] && std::string(_er)!="20" && std::string(_er)!="v5")
          { fprintf(stderr, "[layer] FATAL: unknown TK_REPAIR=%s (want 20|v5)\n", _er); exit(87); } }
    // only for statistic analysis, and thus is not included in runtime
    sum_of_distances = 0;
    for (const auto & agent : agents)
    {
        sum_of_distances += agent.path_planner->my_heuristic[agent.path_planner->start_location];
    }

    initial_solution_runtime = 0;
    start_time = Time::now();
    bool succ = getInitialSolution();
    initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
    if (!succ && initial_solution_runtime < time_limit)
    {
        if (use_init_lns)
        {
            init_lns = new InitLNS(instance, agents, time_limit - initial_solution_runtime,
                    replan_algo_name,init_destroy_name, neighbor_size, screen, bandit_algorithm_name, numberOfNeighborhoodSizeCandidates);
            succ = init_lns->run();
            if (succ) // accept new paths
            {
                path_table.reset();
                for (const auto & agent : agents)
                {
                    path_table.insertPath(agent.id, agent.path);
                }
                init_lns->clear();
                initial_sum_of_costs = init_lns->sum_of_costs;
                sum_of_costs = initial_sum_of_costs;
            g_amor_best = sum_of_costs; // AMOR_RR_PORT
            }
            initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
        }
        else // use random restart
        {
            while (!succ && initial_solution_runtime < time_limit)
            {
                succ = getInitialSolution();
                initial_solution_runtime = ((fsec)(Time::now() - start_time)).count();
                restart_times++;
            }
        }
    }

    int searchSuccess = succ? 1 : 0;
    string weights = "";
    iteration_stats.emplace_back(neighbor.agents.size(),
                                 initial_sum_of_costs, initial_solution_runtime, init_algo_name, weights , 0, 0, searchSuccess);
    runtime = initial_solution_runtime;
    if (succ)
    {
        if (screen >= 1)
            cout << "Initial solution cost = " << initial_sum_of_costs << ", "
                 << "runtime = " << initial_solution_runtime << endl;
    }
    else
    {
        cout << "Failed to find an initial solution in "
             << runtime << " seconds after  " << restart_times << " restarts" << endl;
        return false; // terminate because no initial solution is found
    }

    // ---- ANCHORFIX_PORT: anchor + best-return init at the initial solution (fix A + fix B) ----
    // All init paths (direct success / random restart / init-LNS rescue) converge here with
    // sum_of_costs == initial solution cost; idempotent with the spsa init below (same value).
    if (_amor_on) {
        g_amor_best = sum_of_costs;                                          // ANCHORFIX_PORT (fix A): anchor = initial solution cost, never 1e9
        g_bestret_paths.resize(agents.size());                               // ANCHORFIX_PORT (fix B): initial solution enters best-return tracking
        for (size_t _bri = 0; _bri < agents.size(); _bri++) g_bestret_paths[_bri] = agents[_bri].path;
        if (g_traj_open) { g_traj_os << runtime << "," << g_amor_best << "\n"; g_traj_prev = (long)g_amor_best; }  // ANCHORFIX_PORT: TRAJ initial row
    }

    // ---- AMOR_SPSA_PORT: initialize learned-policy state at the initial solution (spsa only; rr/stock untouched) ----
    if (_tk_spsa) {
        g_amor_best = sum_of_costs;          // clean best init (== initial_sum_of_costs)
        g_sp_wstart = sum_of_costs;
        g_sp_isi = 0; g_sp_wc = 0; g_sp_phase = 0; g_sp_Rplus = 0.0;
        for (int i = 0; i < 4; i++) { g_sp_th[i] = 0.0; g_sp_delta[i] = 1; }
        g_sp_th[3] = _tk_gap_w;              // warm-start GAP weight (default 0 => softplus(0)~0.69 near-greedy)
        g_sp_best_paths.clear();
    }
    if (_tk_cart) { g_cart_vol=0.0; g_cart_prev=sum_of_costs; } // CART_PORT init (runs under cart; moved out of the spsa-only block)

    while (runtime < time_limit && iteration_stats.size() <= num_of_iterations)
    {
        runtime =((fsec)(Time::now() - start_time)).count();
        if(screen >= 1)
            validateSolution();
        if (ALNS)
        {
            chooseDestroyHeuristicbyALNS();
        }
        switch (destroy_strategy)
        {
            case RANDOMWALK:
                succ = generateNeighborByRandomWalk(0);
                break;
            case INTERSECTION:
                succ = generateNeighborByIntersection();
                break;
            case RANDOMAGENTS:
                neighbor.agents.resize(agents.size());
                for (int i = 0; i < (int)agents.size(); i++)
                    neighbor.agents[i] = i;
                if (neighbor.agents.size() > neighbor_size)
                {
                    std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
                    neighbor.agents.resize(neighbor_size);
                }
                assert(neighbor.agents.size() > 0);
                succ = true;
                break;
            case DESTROY_COUNT:
                succ = generateNeighborByRandomWalk(1);
                break;
            default:
                cerr << "Wrong neighbor generation strategy" << endl;
                exit(-1);
        }
        searchSuccess = succ? 1 : 0;
        if(!succ)
            continue;

        // store the neighbor information
        neighbor.old_paths.resize(neighbor.agents.size());
        neighbor.old_sum_of_costs = 0;
        for (int i = 0; i < (int)neighbor.agents.size(); i++)
        {
            if (replan_algo_name == "PP")
                neighbor.old_paths[i] = agents[neighbor.agents[i]].path;
            path_table.deletePath(neighbor.agents[i], agents[neighbor.agents[i]].path);
            neighbor.old_sum_of_costs += agents[neighbor.agents[i]].path.size() - 1;
        }

        if (replan_algo_name == "EECBS")
            succ = runEECBS();
        else if (replan_algo_name == "CBS")
            succ = runCBS();
        else if (replan_algo_name == "PP")
            succ = runPP();
        else
        {
            cerr << "Wrong replanning strategy" << endl;
            exit(-1);
        }
        if (ALNS) // update destroy heuristics
        {
            const bool condition = neighbor.old_sum_of_costs > neighbor.sum_of_costs;
            double value = (neighbor.old_sum_of_costs - neighbor.sum_of_costs);//
            if(neighbor.agents.size())
            {
                value /= neighbor.agents.size();
            }
            updateDestroyAndNeighborhoodWeights(value, condition);
        }
        runtime = ((fsec)(Time::now() - start_time)).count();
        const int _sp_best_before = g_amor_best;                             // AMOR_SPSA_PORT: best before this update
        sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
        if (sum_of_costs < g_amor_best) g_amor_best = sum_of_costs; // AMOR_RR_PORT
        if (g_traj_open && (long)g_amor_best < g_traj_prev) { g_traj_prev = (long)g_amor_best;  // TRAJ_PORT
            g_traj_os << runtime << "," << g_amor_best << "\n"; }
        if (_tk_cart) { // CART_PORT: EMA of |working-cost move| = trajectory volatility
            if (g_cart_prev>=0) { double _d=std::abs((double)sum_of_costs-(double)g_cart_prev);
                g_cart_vol += g_cart_volbeta*(_d - g_cart_vol); }
            g_cart_prev = sum_of_costs;
        }
        if (_amor_on && g_amor_best < _sp_best_before) {                     // ANCHORFIX_PORT (fix C): env-gated // BESTRET_PORT: snapshot best incumbent for UNIVERSAL best-return (all arms)
            g_bestret_paths.resize(agents.size());
            for (size_t _bri = 0; _bri < agents.size(); _bri++) g_bestret_paths[_bri] = agents[_bri].path;
        }
        if (_tk_spsa) {                                                       // AMOR_SPSA_PORT: online learned-policy update (spsa only)
            if (g_amor_best < _sp_best_before) {                             // new global best this iteration
                g_sp_isi = 0;
                g_sp_best_paths.resize(agents.size());                      // snapshot best incumbent for best-return
                for (size_t i = 0; i < agents.size(); i++) g_sp_best_paths[i] = agents[i].path;
            } else {
                g_sp_isi++;
            }
            g_sp_wc++;                                                       // two-window SPSA: R = fractional best-drop over the window
            if (g_sp_wc >= _tk_spsa_W) {
                double R = (double)(g_sp_wstart - g_amor_best) / (double)std::max(1, g_sp_wstart);
                if (R < 0.0) R = 0.0;
                if (g_sp_phase == 0) { g_sp_Rplus = R; g_sp_phase = 1; }    // measured theta + c*Delta; now measure theta - c*Delta
                else {
                    double gsign = (g_sp_Rplus > R) ? 1.0 : ((g_sp_Rplus < R) ? -1.0 : 0.0);
                    for (int i = 0; i < 4; i++) {
                        g_sp_th[i] += _tk_spsa_A * gsign * (double)g_sp_delta[i];
                        if (g_sp_th[i] < -10.0) g_sp_th[i] = -10.0;
                        if (g_sp_th[i] >  10.0) g_sp_th[i] =  10.0;
                    }
                    for (int i = 0; i < 4; i++) g_sp_delta[i] = (rand() % 2) ? 1 : -1; // fresh Rademacher perturbation
                    g_sp_phase = 0;
                }
                g_sp_wc = 0; g_sp_wstart = g_amor_best;
            }
        }
        if (screen >= 1)
            cout << "Iteration " << iteration_stats.size() << ", "
                 << "group size = " << neighbor.agents.size() << ", "
                 << "solution cost = " << sum_of_costs << ", "
                 << "remaining time = " << time_limit - runtime << endl;

        double weightSum = 0;
        for (size_t i = 0; i < neighborhoodBanditStats.size(); ++i) {
            for (int j = 0; j < neighborhoodBanditStats[i]->destroy_weights.size(); j++){
                weightSum += neighborhoodBanditStats[i]->destroy_weights[j];
            }
        }
        iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, replan_algo_name, weights, 0, 0, searchSuccess);
    }

    // ---- AMOR_SPSA_PORT: return the BEST incumbent (spsa accepts uphill => last working solution may be worse) ----
    if (_tk_spsa && !g_sp_best_paths.empty()) {
        for (size_t i = 0; i < agents.size(); i++) agents[i].path = g_sp_best_paths[i];
        sum_of_costs = g_amor_best;
    }

    // ---- BESTRET_PORT: UNIVERSAL best-return (greedy/rr/cart/spsa/rr5). Greedy: working==best => value-identical no-op. ----
    if (_amor_on && !g_bestret_paths.empty()) {   // ANCHORFIX_PORT (fix C): env-gated (stock keeps original last-working-solution semantics)
        for (size_t _bri = 0; _bri < agents.size() && _bri < g_bestret_paths.size(); _bri++) agents[_bri].path = g_bestret_paths[_bri];
        sum_of_costs = g_amor_best;
    }
    if (g_traj_open) { g_traj_os << runtime << "," << g_amor_best << "\n"; g_traj_os.flush(); }  // TRAJ_PORT final
    average_group_size = - iteration_stats.front().num_of_agents;
    for (const auto& data : iteration_stats)
        average_group_size += data.num_of_agents;
    if (average_group_size > 0)
        average_group_size /= (double)(iteration_stats.size() - 1);

    cout << getSolverName() << ": "
         << "runtime = " << runtime << ", "
         << "iterations = " << iteration_stats.size() << ", "
         << "solution cost = " << sum_of_costs << ", "
         << "initial solution cost = " << initial_sum_of_costs << ", "
         << "failed iterations = " << num_of_failures << endl;
    if (_tk_spsa)   // AMOR_SPSA_PORT: log learned policy for mechanism/liveness analysis
        cout << "SPSA_THETA: " << g_sp_th[0] << " " << g_sp_th[1] << " " << g_sp_th[2] << " " << g_sp_th[3]
             << " (isi=" << g_sp_isi << " best=" << g_amor_best << ")" << endl;
    {   // TK_REPAIR_PORT: log arm pulls + EMA values for mechanism/liveness analysis
        const char* _rbs = std::getenv("TK_REPAIR");
        if (_rbs && (atoi(_rbs) == 20 || std::string(_rbs) == "v5")) // REPV5_PORT
            cout << "REPAIR_ARMS n: " << g_rb2_n[0] << " " << g_rb2_n[1] << " " << g_rb2_n[2] << " " << g_rb2_n[3] << " " << g_rb2_n[4]
                 << "  val: " << g_rb2_val[0] << " " << g_rb2_val[1] << " " << g_rb2_val[2] << " " << g_rb2_val[3] << " " << g_rb2_val[4]
                 << "  (arms: random longest shortest most-delayed least-delayed)" << endl;
    }
    return true;
}


bool LNS::getInitialSolution()
{
    neighbor.agents.resize(agents.size());
    for (int i = 0; i < (int)agents.size(); i++)
        neighbor.agents[i] = i;
    neighbor.old_sum_of_costs = MAX_COST;
    neighbor.sum_of_costs = 0;
    bool succ = false;
    if (init_algo_name == "EECBS")
        succ = runEECBS();
    else if (init_algo_name == "PP")
        succ = runPP();
    else if (init_algo_name == "PIBT")
        succ = runPIBT();
    else if (init_algo_name == "PPS")
        succ = runPPS();
    else if (init_algo_name == "winPIBT")
        succ = runWinPIBT();
    else if (init_algo_name == "CBS")
        succ = runCBS();
    else
    {
        cerr <<  "Initial MAPF solver " << init_algo_name << " does not exist!" << endl;
        exit(-1);
    }
    if (succ)
    {
        initial_sum_of_costs = neighbor.sum_of_costs;
        sum_of_costs = neighbor.sum_of_costs;
        return true;
    }
    else
    {
        return false;
    }

}

bool LNS::runEECBS()
{
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    ECBS ecbs(search_engines, screen - 1, &path_table);
    ecbs.setPrioritizeConflicts(true);
    ecbs.setDisjointSplitting(false);
    ecbs.setBypass(true);
    ecbs.setRectangleReasoning(true);
    ecbs.setCorridorReasoning(true);
    ecbs.setHeuristicType(heuristics_type::WDG, heuristics_type::GLOBAL);
    ecbs.setTargetReasoning(true);
    ecbs.setMutexReasoning(false);
    ecbs.setConflictSelectionRule(conflict_selection::EARLIEST);
    ecbs.setNodeSelectionRule(node_selection::NODE_CONFLICTPAIRS);
    ecbs.setSavingStats(false);
    double w;
    if (iteration_stats.empty())
        w = 5; // initial run
    else
        w = 1.1; // replan
    ecbs.setHighLevelSolver(high_level_solver_type::EES, w);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime;
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = ecbs.solve(T, 0);
    if (succ && ecbs.solution_cost < neighbor.old_sum_of_costs) // accept new paths
    {
        auto id = neighbor.agents.begin();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *ecbs.paths[i];
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = ecbs.solution_cost;
        if (sum_of_costs_lowerbound < 0)
            sum_of_costs_lowerbound = ecbs.getLowerBound();
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        if (!succ)
            num_of_failures++;
    }
    return succ;
}
bool LNS::runCBS()
{
    if (screen >= 2)
        cout << "old sum of costs = " << neighbor.old_sum_of_costs << endl;
    vector<SingleAgentSolver*> search_engines;
    search_engines.reserve(neighbor.agents.size());
    for (int i : neighbor.agents)
    {
        search_engines.push_back(agents[i].path_planner);
    }

    CBS cbs(search_engines, screen - 1, &path_table);
    cbs.setPrioritizeConflicts(true);
    cbs.setDisjointSplitting(false);
    cbs.setBypass(true);
    cbs.setRectangleReasoning(true);
    cbs.setCorridorReasoning(true);
    cbs.setHeuristicType(heuristics_type::WDG, heuristics_type::ZERO);
    cbs.setTargetReasoning(true);
    cbs.setMutexReasoning(false);
    cbs.setConflictSelectionRule(conflict_selection::EARLIEST);
    cbs.setNodeSelectionRule(node_selection::NODE_CONFLICTPAIRS);
    cbs.setSavingStats(false);
    cbs.setHighLevelSolver(high_level_solver_type::ASTAR, 1);
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime; // time limit
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    bool succ = cbs.solve(T, 0);
    if (succ && cbs.solution_cost <= neighbor.old_sum_of_costs) // accept new paths
    {
        auto id = neighbor.agents.begin();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *cbs.paths[i];
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = cbs.solution_cost;
        if (sum_of_costs_lowerbound < 0)
            sum_of_costs_lowerbound = cbs.getLowerBound();
    }
    else // stick to old paths
    {
        if (!neighbor.old_paths.empty())
        {
            for (int id : neighbor.agents)
            {
                path_table.insertPath(agents[id].id, agents[id].path);
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;

        }
        if (!succ)
            num_of_failures++;
    }
    return succ;
}
bool LNS::runPP()
{
    static const int    _sata_sa   = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="sa")?1:0; }(); // SATA_PORT: simulated annealing
    static const int    _sata_ta   = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="ta")?1:0; }(); // SATA_PORT: threshold accepting
    static const double _sata_t0   = [](){ const char* s=std::getenv("TK_SA_T0");   return s?atof(s):20.0; }();  // SATA_PORT: SA initial temperature
    static const double _sata_tau0 = [](){ const char* s=std::getenv("TK_TA_TAU0"); return s?atof(s):20.0; }();  // SATA_PORT: TA initial threshold
    static const int _tk_rr = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="rr")?1:0; }();
    static const int _tk_delta = [](){ const char* s=std::getenv("TK_RR_DELTA"); return s?atoi(s):0; }();
    static const int    _tk_spsa       = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="spsa")?1:0; }(); // AMOR_SPSA_PORT
    static const double _tk_spsa_snorm = [](){ const char* s=std::getenv("TK_SPSA_SNORM"); return s?atof(s):50.0; }();
    static const double _tk_spsa_c     = [](){ const char* s=std::getenv("TK_SPSA_C");     return s?atof(s):0.6; }();
    static const int _tk_cart = [](){ const char* s=std::getenv("TK_ACCEPT"); return (s && std::string(s)=="cart")?1:0; }(); // CART_PORT
    static const double _tk_cart_d0 = [](){ const char* s=std::getenv("TK_CART_D0"); return s?atof(s):10.0; }();
    static const int _tk_cart_vol = [](){ const char* s=std::getenv("TK_CART_MODE"); return (s && std::string(s)=="vol")?1:0; }();
    static const double _tk_cart_k = [](){ const char* s=std::getenv("TK_CART_K"); return s?atof(s):1.0; }();
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
    static const int _tk_rb = [](){ const char* s=std::getenv("TK_REPAIR"); return s?atoi(s):0; }(); // TK_REPAIR_PORT
    static const int _tk_rbv5 = [](){ const char* s=std::getenv("TK_REPAIR"); return (s && std::string(s)=="v5")?1:0; }(); // REPV5_PORT
    if ((_tk_rb == 20 || _tk_rbv5) && !iteration_stats.empty() && !neighbor.old_paths.empty()) // REPV5_PORT
    {   // TK_REPAIR_PORT: replan-only arm pull -- pick a priority rule, re-order shuffled_agents accordingly.
        rb2_cfg(); if (_tk_rbv5) ts_cfg();
        int _arm = -1;
        for (int a = 0; a < 5; a++) if (g_rb2_allow[a] && g_rb2_n[a] == 0) { _arm = a; break; }  // warmup: each allowed arm once
        if (_arm < 0 && _tk_rbv5) {   // REPV5_PORT: Thompson sampling over per-arm Beta posteriors (commits ~proportional to certainty)
            double _best = -1.0;
            for (int a = 0; a < 5; a++) if (g_rb2_allow[a]) { double _th = ts_beta(g_ts_a[a], g_ts_b[a]); if (_th > _best) { _best = _th; _arm = a; } }
        }
        if (_arm < 0) {
            if ((double)rand()/RAND_MAX < g_rb2_eps) {          // explore uniformly among allowed arms
                int _pool[5], _np = 0; for (int a = 0; a < 5; a++) if (g_rb2_allow[a]) _pool[_np++] = a;
                _arm = _pool[rand() % std::max(1, _np)];
            } else {                                            // exploit: best EMA value
                for (int a = 0; a < 5; a++) if (g_rb2_allow[a] && (_arm < 0 || g_rb2_val[a] > g_rb2_val[_arm])) _arm = a;
            }
        }
        if (_arm < 0) _arm = 0;
        g_rb2_last = _arm; g_rb2_lb = 0;
        for (int id : shuffled_agents) g_rb2_lb += agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location];
        auto _hd = [&](int a) { return (int)agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]; };
        auto _dd = [&](int a) { return (int)agents[a].path.size() - 1 - _hd(a); };
        if      (_arm == 1) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a, int b){ return _hd(a) > _hd(b); });
        else if (_arm == 2) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a, int b){ return _hd(a) < _hd(b); });
        else if (_arm == 3) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a, int b){ return _dd(a) > _dd(b); });
        else if (_arm == 4) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a, int b){ return _dd(a) < _dd(b); });
        // _arm == 0: keep the stock random_shuffle above (anchor arm == original behavior)
    }
    if (screen >= 2) {
        for (auto id : shuffled_agents)
            cout << id << "(" << agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location] <<
                "->" << agents[id].path.size() - 1 << "), ";
        cout << endl;
    }
    int remaining_agents = (int)shuffled_agents.size();
    auto p = shuffled_agents.begin();
    neighbor.sum_of_costs = 0;
    runtime = ((fsec)(Time::now() - start_time)).count();
    double T = time_limit - runtime; // time limit
    if (!iteration_stats.empty()) // replan
        T = min(T, replan_time_limit);
    auto time = Time::now();
    ConstraintTable constraint_table(instance.num_of_cols, instance.map_size, &path_table);
    while (p != shuffled_agents.end() && ((fsec)(Time::now() - time)).count() < T)
    {
        int id = *p;
        if (screen >= 3)
            cout << "Remaining agents = " << remaining_agents <<
                 ", remaining time = " << T - ((fsec)(Time::now() - time)).count() << " seconds. " << endl
                 << "Agent " << agents[id].id << endl;
        agents[id].path = agents[id].path_planner->findPath(constraint_table);
        if (agents[id].path.empty()) break;
        neighbor.sum_of_costs += (int)agents[id].path.size() - 1;
        if (!_tk_rr && !_tk_spsa && !_tk_cart && !_sata_sa && !_sata_ta && neighbor.sum_of_costs >= neighbor.old_sum_of_costs) // SATA_PORT // CART_PORT // AMOR_SPSA_PORT: disable greedy early-break under spsa too
            break;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
     if ((destroy_strategy == RANDOMWALK || destroy_strategy == DESTROY_COUNT) && (algo != CANONICAL)){
        bool no_tackle_small = algo != TACKLE_SMALL && current_index >= 0;
        bool tackle_small = algo == TACKLE_SMALL && counterfactual_index >= 0;
        if (no_tackle_small || tackle_small){
            int suc = (neighbor.old_sum_of_costs - neighbor.sum_of_costs) > 0 ? 1 : -1;
            if (algo == BERNOULIE|| destroy_strategy == DESTROY_COUNT){
                if (suc == 1){
                    alpha[current_index]++;
                } else {
                    beta[current_index]++;
                }
            }
            if (algo == TACKLE_SMALL){
                if(counterfactual_index >= 0 && intent_index >= 0){
                    if (suc == 1){
                        counterfactual_alpha[intent_index][counterfactual_index]++;
                    } else {
                        counterfactual_beta[intent_index][counterfactual_index]++;
                    }
                }
            }
            //for normal distribution
            else if (algo == NORMAL){
                double reward = sum_of_costs;
                double n = (*frequency)[current_index];
                double old_mu = mu[current_index];
                double old_sigma2 = sigma2[current_index];

                // Update mean and variance
                double new_mu = old_mu + (reward - old_mu) / n;
                double new_sigma2 = ((n - 1) * old_sigma2 + (reward - old_mu) * (reward - new_mu)) / n;

                mu[current_index] = new_mu;
                sigma2[current_index] = new_sigma2;
            }
        } 
    } else if (destroy_strategy == INTERSECTION){
        if (algo != CANONICAL && current_index != -1){
            //update location q values
            //outFile << "old: " << neighbor.old_sum_of_costs << " new : " << neighbor.sum_of_costs << endl;
            double sum = ((*location_q_values)[current_index] * ((*location_frequency)[current_index] - 1)) + ((neighbor.old_sum_of_costs - neighbor.sum_of_costs));
            double temp = (sum / (*location_frequency)[current_index]);
            (*location_q_values)[current_index] = temp;
            if (algo == BERNOULIE){
                int suc = (neighbor.old_sum_of_costs - neighbor.sum_of_costs) > 0 ? 1 : -1;
                if (suc == 1){
                    location_alpha[current_index]++;
                } else {
                    location_beta[current_index]++;
                }
            }
            else if (algo == NORMAL){
                double reward = sum_of_costs;
                double n = (*location_frequency)[current_index];
                double old_mu = location_mu[current_index];
                double old_sigma2 = location_sigma2[current_index];

                // Update mean and variance
                double new_mu = old_mu + (reward - old_mu) / n;
                double new_sigma2 = ((n - 1) * old_sigma2 + (reward - old_mu) * (reward - new_mu)) / n;

                location_mu[current_index] = new_mu;
                location_sigma2[current_index] = new_sigma2;
            }
        }
    }
    int _rr_allow; // AMOR_RR_PORT / AMOR_SPSA_PORT
    if (_sata_sa) {                                                          // SATA_PORT: simulated annealing (current-based Metropolis)
        int _delta = neighbor.sum_of_costs - neighbor.old_sum_of_costs;      // Delta = candidate cost - current accepted cost
        bool _acc;
        if (remaining_agents != 0)      _acc = false;                        // broken/partial replan => reject (also AND-guarded below)
        else if (_delta <= 0)           _acc = true;                         // downhill / lateral => always accept
        else {
            double _frac = (time_limit>0.0) ? (1.0 - runtime/time_limit) : 0.0; if (_frac<0.0) _frac=0.0;
            double _temp = _sata_t0 * _frac; if (_temp < 1e-6) _temp = 1e-6; // linear cooling to ~0 (floored to avoid div0)
            double _pacc = std::exp(-(double)_delta / _temp);
            _acc = (g_sata_u01() < _pacc);                                   // Metropolis draw from the SEPARATE RNG
        }
        _rr_allow = _acc ? _delta : (_delta - 1);                           // encode the boolean decision into the shared integer test
    }
    else if (_sata_ta) {                                                     // SATA_PORT: threshold accepting (current-based, deterministic)
        double _frac = (time_limit>0.0) ? (1.0 - runtime/time_limit) : 0.0; if (_frac<0.0) _frac=0.0;
        double _tau = _sata_tau0 * _frac;                                    // linear threshold decay to 0
        _rr_allow = (int)(_tau + 0.5);                                       // accept iff Delta <= round(tau); round-half to integer SoC grid
    }
    else
    if (_tk_spsa) {                                                          // learned delta_eff = softplus(theta . [1, s, u, g]) w/ SPSA perturbation
        double s_feat = std::min(1.0, (double)g_sp_isi / _tk_spsa_snorm);
        double u_feat = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
        double _lb = (double)sum_of_distances;                              // TACKLE SoC lower bound = sum of individual shortest-path distances
        double _init_gap = std::max(1.0, (double)initial_sum_of_costs - _lb);
        double g_feat = ((double)g_amor_best - _lb) / _init_gap;
        if (g_feat < 0.0) g_feat = 0.0; if (g_feat > 1.0) g_feat = 1.0;
        double sign = (g_sp_phase == 0) ? 1.0 : -1.0;                       // act with theta +/- c*Delta (two-window SPSA)
        double t0 = g_sp_th[0] + sign * _tk_spsa_c * g_sp_delta[0];
        double t1 = g_sp_th[1] + sign * _tk_spsa_c * g_sp_delta[1];
        double t2 = g_sp_th[2] + sign * _tk_spsa_c * g_sp_delta[2];
        double t3 = g_sp_th[3] + sign * _tk_spsa_c * g_sp_delta[3];
        double z  = t0 + t1 * s_feat + t2 * u_feat + t3 * g_feat;
        double de = (z > 30.0) ? z : std::log(1.0 + std::exp(z));           // softplus (overflow-guarded)
        if (de < 0.0) de = 0.0;
        int dei = (int)(de + 0.5);                                          // round to int
        _rr_allow = g_amor_best + dei - sum_of_costs;
    } else if (_tk_cart) {                                                  // CART_PORT: budget-decayed record-to-record
        double _frac = (time_limit>0.0) ? (1.0 - runtime/time_limit) : 0.0; if (_frac<0.0) _frac=0.0;
        double _dev = _tk_cart_d0 * _frac;
        if (_tk_cart_vol) { double _vn = g_cart_vol / std::max(1.0,(double)(initial_sum_of_costs - sum_of_distances)/std::max(1,(int)agents.size()));
                          _dev = _tk_cart_d0 * _tk_cart_k * _vn * _frac; }
        int _cdei = (int)(_dev + 0.5);                                          // L1 fix: round-half to the integer SoC-tolerance grid (no floor dead-zone; matches spsa dei)
        _rr_allow = g_amor_best + _cdei - sum_of_costs;
    } else {
        _rr_allow = _tk_rr ? (g_amor_best + _tk_delta - sum_of_costs) : 0;  // AMOR_RR_PORT (unchanged)
    }
    if (g_rb2_last >= 0)   // TK_REPAIR_PORT: credit the pulled arm on EVERY outcome (accept, reject, failure)
    {
        bool _rb_acc = (remaining_agents == 0 && neighbor.sum_of_costs <= neighbor.old_sum_of_costs + _rr_allow);
        if (_tk_rbv5) {       // REPV5_PORT: magnitude-aware Bernoulli reward + Beta-TS posterior update
            double _p = 0.0;
            if (_rb_acc) { _p = ((double)neighbor.old_sum_of_costs - (double)neighbor.sum_of_costs) / g_ts_scale;
                if (_p < 0.0) _p = 0.0; if (_p > 1.0) _p = 1.0; }
            double _succ = (ts_u01() < _p) ? 1.0 : 0.0;   // randomized rounding of the raw SoC delta
            g_ts_a[g_rb2_last] += _succ; g_ts_b[g_rb2_last] += (1.0 - _succ);
            g_rb2_n[g_rb2_last]++;
            g_rb2_val[g_rb2_last] = g_ts_a[g_rb2_last] / (g_ts_a[g_rb2_last] + g_ts_b[g_rb2_last]); // log: posterior mean
        } else {
        double _rb_r = 0.0;   // reject/failure => neighborhood reverts to old paths => reward 0 (arm failure rate is learned)
        if (_rb_acc) {        // slack-fraction reward (fork-identical): (old-new)/max(1, old-lb), clipped to [-1,1]
            double _sl = std::max(1.0, (double)neighbor.old_sum_of_costs - g_rb2_lb);
            _rb_r = ((double)neighbor.old_sum_of_costs - (double)neighbor.sum_of_costs) / _sl;
            if (_rb_r < -1.0) _rb_r = -1.0; if (_rb_r > 1.0) _rb_r = 1.0;
        }
        g_rb2_val[g_rb2_last] += g_rb2_alpha * (_rb_r - g_rb2_val[g_rb2_last]); g_rb2_n[g_rb2_last]++;
        }
        g_rb2_last = -1;
    }
    if (remaining_agents == 0 && neighbor.sum_of_costs <= neighbor.old_sum_of_costs + _rr_allow) // accept new paths
    {
        return true;
    }
    else // stick to old paths
    {
        if (p != shuffled_agents.end())
            num_of_failures++;
        auto p2 = shuffled_agents.begin();
        while (p2 != p)
        {
            int a = *p2;
            path_table.deletePath(agents[a].id, agents[a].path);
            ++p2;
        }
        if (!neighbor.old_paths.empty())
        {
            p2 = neighbor.agents.begin();
            for (int i = 0; i < (int)neighbor.agents.size(); i++)
            {
                int a = *p2;
                agents[a].path = neighbor.old_paths[i];
                path_table.insertPath(agents[a].id, agents[a].path);
                ++p2;
            }
            neighbor.sum_of_costs = neighbor.old_sum_of_costs;
        }
        return false;
    }
}
bool LNS::runPPS(){
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);
    P.setTimestepLimit(pipp_option.timestepLimit);

    // seed for solver
    auto* MT_S = new std::mt19937(0);
    PPS solver(&P,MT_S);
    solver.setTimeLimit(time_limit);
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}
bool LNS::runPIBT(){
    auto shuffled_agents = neighbor.agents;
     std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);

    // seed for solver
    auto MT_S = new std::mt19937(0);
    PIBT solver(&P,MT_S);
    solver.setTimeLimit(time_limit);
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}
bool LNS::runWinPIBT(){
    auto shuffled_agents = neighbor.agents;
    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());

    MAPF P = preparePIBTProblem(shuffled_agents);
    P.setTimestepLimit(pipp_option.timestepLimit);

    // seed for solver
    auto MT_S = new std::mt19937(0);
    winPIBT solver(&P,pipp_option.windowSize,pipp_option.winPIBTSoft,MT_S);
    solver.setTimeLimit(time_limit);
    bool result = solver.solve();
    if (result)
        updatePIBTResult(P.getA(),shuffled_agents);
    return result;
}

MAPF LNS::preparePIBTProblem(vector<int>& shuffled_agents){

    // seed for problem and graph
    auto MT_PG = new std::mt19937(0);
    Graph* G = new SimpleGrid(instance.getMapFile());

    std::vector<Task*> T;
    PIBT_Agents A;

    for (int i : shuffled_agents){
        assert(G->existNode(agents[i].path_planner->start_location));
        assert(G->existNode(agents[i].path_planner->goal_location));
        auto a = new PIBT_Agent(G->getNode( agents[i].path_planner->start_location));
        A.push_back(a);
        Task* tau = new Task(G->getNode( agents[i].path_planner->goal_location));


        T.push_back(tau);
        if(screen>=5){
            cout<<"Agent "<<i<<" start: " <<a->getNode()->getPos()<<" goal: "<<tau->getG().front()->getPos()<<endl;
        }
    }

    return MAPF(G, A, T, MT_PG);

}

void LNS::updatePIBTResult(const PIBT_Agents& A, vector<int>& shuffled_agents){
    int soc = 0;
    for (int i=0; i<A.size();i++){
        int a_id = shuffled_agents[i];

        agents[a_id].path.resize(A[i]->getHist().size());
        int last_goal_visit = 0;
        if(screen>=2)
            std::cout<<A[i]->logStr()<<std::endl;
        for (int n_index = 0; n_index < A[i]->getHist().size(); n_index++){
            auto n = A[i]->getHist()[n_index];
            agents[a_id].path[n_index] = PathEntry(n->v->getId());

            //record the last time agent reach the goal from a non-goal vertex.
            if(agents[a_id].path_planner->goal_location == n->v->getId()
                && n_index - 1>=0
                && agents[a_id].path_planner->goal_location !=  agents[a_id].path[n_index - 1].location)
                last_goal_visit = n_index;

        }
        //resize to last goal visit time
        agents[a_id].path.resize(last_goal_visit + 1);
        if(screen>=2)
            std::cout<<" Length: "<< agents[a_id].path.size() <<std::endl;
        if(screen>=5){
            cout <<"Agent "<<a_id<<":";
            for (auto loc : agents[a_id].path){
                cout <<loc.location<<",";
            }
            cout<<endl;
        }
        path_table.insertPath(agents[a_id].id, agents[a_id].path);
        soc += (int)agents[a_id].path.size()-1;
    }

    neighbor.sum_of_costs =soc;
}

void LNS::chooseDestroyHeuristicbyALNS()
{
    sampleDestroyHeuristicAndNeighborhoodSize();
    if (alns_bernoulie == BCANONICAL ){
        switch (selected_neighbor)
        {
            case 0 : destroy_strategy = RANDOMWALK;cout << "randomwalk" << endl;break;
            case 1 : destroy_strategy = INTERSECTION; cout << "intersection" << endl;break;
            case 2 : destroy_strategy = RANDOMAGENTS; cout << "random" << endl;break;
            default : cerr << "ERROR" << endl; exit(-1);
        }
    }
    else if (alns_bernoulie == REPLACE){
        switch (selected_neighbor)
        {
            // Replace random walk with bernoulie
            case 0 : destroy_strategy = DESTROY_COUNT;cout << "bernoulie" << endl;break;
            case 1 : destroy_strategy = INTERSECTION; cout << "intersection" << endl;break;
            case 2 : destroy_strategy = RANDOMAGENTS; cout << "random" << endl;break;
            default : cerr << "ERROR" << endl; exit(-1);
        }
    }
    else {
        switch (selected_neighbor)
        {
            // Add bernoulie as the 4th option
            case 0 : destroy_strategy = RANDOMWALK;cout << "randomwalk" << endl;break;
            case 1 : destroy_strategy = INTERSECTION; cout << "intersection" << endl;break;
            case 2 : destroy_strategy = RANDOMAGENTS; cout << "random" << endl;break;
            case 3 : destroy_strategy = DESTROY_COUNT; cout << "bernoulie" << endl;break;
            default : cerr << "ERROR" << endl; exit(-1);
        }
    }
    
}

bool LNS::generateNeighborByIntersection()
{   
    
    set<int> neighbors_set;
    if (intersections.empty())
    {
        for (int i = 0; i < instance.map_size; i++)
        {
            if (!instance.isObstacle(i) && instance.getDegree(i) > 2)
                intersections.push_back(i);
                num_valid_spaces++;
        }
        for (int i = 0; i < regions; i++){
            (*location_q_values).push_back(INT_MAX-1);
            (*location_frequency).push_back(0); 
            (location_mu).push_back(0.0);
            (location_sigma2).push_back(1.0);
        }
    }
    int location = 0;
    // The wrapper dynamically chosses the bandit implementation:
    int region = location_wrapper();
    if (region == -1){
        auto pt = intersections.begin();
        std::advance(pt, rand() % intersections.size());
        location = *pt;
    } else {
        int start = region * (num_valid_spaces / regions);
        int end = start + (num_valid_spaces / regions);
        auto pt = intersections.begin();
        int num_step = 20;
        std::advance(pt, start+(rand() % end));
        location = *pt;
    }
    

    path_table.get_agents(neighbors_set, neighbor_size, location);
    if (neighbors_set.size() < neighbor_size)
    {
        set<int> closed;
        closed.insert(location);
        std::queue<int> open;
        open.push(location);
        while (!open.empty() && (int) neighbors_set.size() < neighbor_size)
        {
            int curr = open.front();
            open.pop();
            for (auto next : instance.getNeighbors(curr))
            {
                if (closed.count(next) > 0)
                    continue;
                open.push(next);
                closed.insert(next);
                if (instance.getDegree(next) >= 3)
                {
                    path_table.get_agents(neighbors_set, neighbor_size, next);
                    if ((int) neighbors_set.size() == neighbor_size)
                        break;
                }
            }
        }
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (neighbor.agents.size() > neighbor_size)
    {
        std::random_shuffle(neighbor.agents.begin(), neighbor.agents.end());
        neighbor.agents.resize(neighbor_size);
    }
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by intersection " << location << endl;
    return true;
}

bool LNS::generateNeighborByRandomWalk(int b)
{
    if (neighbor_size >= (int)agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }
    int a = -1;
    if (b){
        a = bernoulie();
    }
    else {
        a = wrapper();
    }
    if (a < 0) {
        return false;
    }
    
    set<int> neighbors_set;
    neighbors_set.insert(a);
    randomWalk(a, agents[a].path[0].location, 0, neighbors_set, neighbor_size, (int) agents[a].path.size() - 1);
    int count = 0;
    while (neighbors_set.size() < neighbor_size && count < 10)
    {
        int t = rand() % agents[a].path.size();
        randomWalk(a, agents[a].path[t].location, t, neighbors_set, neighbor_size, (int) agents[a].path.size() - 1);
        count++;
        // select the next agent randomly
        int idx = rand() % neighbors_set.size();
        int i = 0;
        for (auto n : neighbors_set)
        {
            if (i == idx)
            {
                a = i;
                break;
            }
            i++;
        }
    }
    if (neighbors_set.size() < 2){
        return false;
    }
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by random walks of agent " << a
             << "(" << agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]
             << "->" << agents[a].path.size() - 1 << ")" << endl;
    return true;
}

/**
 * @brief ADDRESS: a wrapper for various bandit sampeling methods. 
 * Returns the selected agent index and sets global variables. 
 * 
 * @return int 
 */
int LNS::wrapper(){
    switch(algo){
        case CANONICAL:
            return findMostDelayedAgent();
            break;
        case GREEDY:
            return greedy();
            break;
        case RANDOM_SEED:
            return random();
            break;
        case ROULETTE_SEED:
            return roulette();
            break;
        case EPSILON:
            return epsilonGreedy();
            break;
        case EPSILON_DECAY:
            return decayEpsilonGreedy();
            break;
        case UCB:
            return ucb();
            break;
        case TOPK_GREEDY:
            return topKEpsilonGreedy();
            break;
        case TOPK_EPSILON:
            return topKEpsilonGreedy();
            break;
        case TOP_EPSILON_DECAY:
            return topKDecayEpsilonGreedy();
            break;
        case TOPK_UCB:
            return topKUCB();
            break;
        case BERNOULIE :
            return bernoulie();
            break;
        case TACKLE_SMALL :
            return tackle_small();
            break;
        case NORMAL :
            return normal();
            break;
    }
}

/**
 * @brief ADDRESS: wrapper for location-based various Bandit sampeling methods.
 * Returns an integer indicating the region paritition that is selected. 
 * 
 * @return int 
 */
int LNS::location_wrapper(){
    switch(algo){
        case CANONICAL:
            return -1;
            break;
        case GREEDY:
            return location_greedy();
            break;
        case EPSILON:
            return location_epsilonGreedy();
            break;
        case EPSILON_DECAY:
            return location_decay_epsilonGreedy();
            break;
        case UCB:
            return location_UCB();
            break;
        case BERNOULIE:
            return location_bernoulie();
            break;
        case NORMAL:
            return location_normal();
            break;
        default:
            cout << "Error algorithm not implemented" << endl;
            exit(0);
    }
}

/**
 * @brief ADDRESS: location-based greedy bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::location_greedy(){
    double min = INT_MAX;
    int index = 0;
    for (int i =0 ; i < location_q_values->size(); i++){
        if ((*location_q_values)[i] < min ){
            index = i;
            min = (*location_q_values)[i];
        }
    }
    (*location_frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: location-based epsilon-greedy bandit sampeling implementation. 
 * 
 * @return int 
 */
int LNS::location_epsilonGreedy(){
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //if the first selection or epsilon explore
    if (dist(rng) < epsilon) {
        //random arm 
        std::uniform_int_distribution<int> dist2(0, location_q_values->size() - 1);
        int rand = dist2(rng);
        (*location_frequency)[rand]++;
        current_index = rand;
        return rand;
    }
    else{
        //best estimate 
        double max = INT_MIN;
        int index = 0;
        for (int i =0 ; i < location_q_values->size(); i++){
            if ((*location_q_values)[i] > max ){
                index = i;
                max = (*location_q_values)[i];
            }
        }
        (*location_frequency)[index]++;
        current_index = index;
        return index;
    }
}

/**
 * @brief ADDRESS: location-based decay-epsilon-greedy bandit sampeling implementation
 * 
 * @return int 
 */
int LNS::location_decay_epsilonGreedy(){
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (epsilon >= 0){
        epsilon -= decay;
    }
    //if the first selection or epsilon explore
    if (dist(rng) < epsilon|| std::all_of(location_frequency->begin(), location_frequency->end(), [](int i){ return i == 0; })) {
        //random arm 
        std::uniform_int_distribution<int> dist2(0, location_q_values->size() - 1);
        int rand = dist2(rng);
        (*location_frequency)[rand]++;
        current_index = rand;
        return rand;
    }
    else{
        //best estimate 
        double max = INT_MIN;
        int index = 0;
        for (int i =0 ; i < location_q_values->size(); i++){
            if ((*location_q_values)[i] >= max ){
                index = i;
                max = (*location_q_values)[i];
            }
        }
        (*location_frequency)[index]++;
        current_index = index;
        return index;
    }
}

/**
 * @brief ADDRESS: location-based bernoulie bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::location_bernoulie(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> samples(regions);
    // Sample from the Beta distribution for each agent
    for (int i = 0 ; i < regions ; i++) {
        boost::random::beta_distribution<> beta_dist(location_alpha[i], location_beta[i]);
        samples[i] = beta_dist(gen);
    }

    // Find the agent with the highest sample value
    int index = std::distance(samples.begin(), std::max_element(samples.begin(), samples.end()));
    (*location_frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: location priority-queue comparator implementation.
 * 
 */
struct LocationComparator{
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        return a.first < b.first;
    }
};

int LNS::findMostDelayedAgent()
{
    int a = -1;
    current_index = a;
    int max_delays = -1;
    for (int i = 0; i < agents.size(); i++)
    {
        if (tabu_list.find(i) != tabu_list.end())
            continue;
        int delays = agents[i].getNumOfDelays();
        if (max_delays < delays)
        {
            a = i;
            max_delays = delays;
        }
    }
    if (max_delays == 0)
    {
        tabu_list.clear();
        return -1;
    }
    tabu_list.insert(a);
    if (tabu_list.size() == agents.size())
        tabu_list.clear();
    current_index = a;
    return a;
}

/**
 * @brief agent-based epsilon greedy bandit sampling implementation. 
 * 
 * @return int 
 */
int LNS::epsilonGreedy()
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //if the first selection or epsilon explore
        if (dist(rng) < epsilon || std::all_of(frequency->begin(), frequency->end(), [](int i){ return i == 0; })) {
            //random arm 
            std::uniform_int_distribution<int> dist2(0, q_values->size() - 1);
            int rand = dist2(rng);
            (*frequency)[rand]++;
            current_index = rand;
            return rand;
        }
        else{
            //best estimate 
            int min = INT_MAX;
            int index = 0;
            for (int i =0 ; i < q_values->size(); i++){
                if ((*q_values)[i] < min ){
                    index = i;
                    min = (*q_values)[i];
                }
            }
            (*frequency)[index]++;
            current_index = index;
            return index;
        }
}

/**
 * @brief ADDRESS: agent-based priority-queue comparator.
 * 
 */
struct AgentComparator {
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const {
        return a.first < b.first;
    }
};

/**
 * @brief ADDRESS: agent-based greedy bandit sampling implementation.
 * 
 * @return int 
 */
int LNS::greedy()
{
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }
    return queue.top().second;
}

int LNS::random()
{
    return rand() % num_agent;
}

int LNS::roulette()
{
    double total_delay = sum_of_costs - sum_of_costs_lowerbound;
    //build the top k of most delayed
    double r = (double) rand() / RAND_MAX;
    double threshold = 0;
    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        threshold += delay;
        int id = agent.id;
        if(threshold >= r*total_delay) {
            return id;
        }
    }
    return rand() % num_agent;
}

/**
 * @brief ADDRESS: agent-based bernoulie bandit sampeling implementation.
 * This Bernoulie Bandit is the choice bandit for ADDRESS. 
 * 
 * @return int 
 */
int LNS::bernoulie() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> samples(k);
    //build the top k of most delayed
    
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }

    // Extract the top 10 agents
    std::vector<int> topAgents;
    for (int i = 0; i < k && !queue.empty(); i++) {
        topAgents.push_back(queue.top().second);
        queue.pop();
    }
    // Sample from the Beta distribution for each agent
    int counter = 0;
    for (int i : topAgents) {
        boost::random::beta_distribution<> beta_dist(alpha[i], beta[i]);
        samples[counter] = beta_dist(gen);
        counter++;
    }

    // Find the agent with the highest sample value
    int best_index = std::distance(samples.begin(), std::max_element(samples.begin(), samples.end()));
    int best_agent = topAgents[best_index];
    (*frequency)[best_agent]++;
    current_index = best_agent;
    return best_agent;
}

int LNS::tackle_small() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> samples(k);
    static double shortest_path_distance_sum = -1;
    if(shortest_path_distance_sum < 0) {
        shortest_path_distance_sum = 0;
        for (const Agent& agent : agents) {
            shortest_path_distance_sum += agent.path_planner->my_heuristic[agent.path_planner->start_location];
        }
    }
    double total_delay = sum_of_costs - shortest_path_distance_sum;
    //build the top k of most delayed
    double r = (double) rand() / RAND_MAX;
    double threshold = 0;
    int intent_candidate = -1;
    
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
        threshold += delay;
        if(intentStrategy == intent_strategy::ROULETTE_INTENT_BASED && intent_candidate < 0 && threshold >= r*total_delay) {
            intent_candidate = id;
        }
    }
    if(intentStrategy == intent_strategy::ROULETTE_INTENT_BASED && intent_candidate < 0) {
        intent_candidate = num_agent;
    }

    if(intentStrategy == intent_strategy::GREEDY_INTENT_BASED) {
        intent_candidate = queue.top().second;
    }

    if(intentStrategy == intent_strategy::RANDOM_INTENT_BASED) {
        intent_candidate = rand() % num_agent;
    }
    int max_delays = queue.top().first;
    // Extract the top 10 agents
    static std::vector<int> topAgents(k, -1);
    bool stateChanged = false;
    intent_index = -1;
    for (int i = 0; i < k; i++) {
        int agentId;
        if(queue.empty()) {
            agentId = -1;
        } else {
            agentId = queue.top().second;
            queue.pop();
        }
        stateChanged = stateChanged || agentId != topAgents[i];
        topAgents[i] = agentId;
        if(intent_candidate == agentId) {
            intent_index = i;
        }
        if(intentStrategy == intent_strategy::TABU_INTENT_BASED && tabu_list.count(agentId) == 0) {
            intent_index = i;
            tabu_list.insert(agentId);
            tabuListCounter += 1;
        }
    }
    if(intentStrategy == intent_strategy::CONSTANT_INTENT_BASED) {
        intent_index = 0;
    }
    if(intent_index < 0 && intentStrategy == intent_strategy::TABU_INTENT_BASED) {
        tabuListCounter += 1;
        if(tabuListCounter >= num_agent) {
            tabu_list.clear();
        }
    }

    if(nonStationaryBandits) {
        if(stateChanged) {
            for(int i = 0; i < k+1; i++) {
                std::fill(counterfactual_alpha[i].begin(), counterfactual_alpha[i].end(), 1);
                std::fill(counterfactual_beta[i].begin(), counterfactual_beta[i].end(), 1);
            }
        }
        // Sample from the Beta distribution for each agent
        int counter = 0;
        if(intent_index >= 0) {
            for (int i : topAgents) {
                boost::random::beta_distribution<> beta_dist(counterfactual_alpha[intent_index][counter], counterfactual_beta[intent_index][counter]);
                samples[counter] = beta_dist(gen);
                counter++;
            }
        }
    } else {
        if(intent_index >= 0) {
            // Sample from the Beta distribution for each agent
            int counter = 0;
            for (int i : topAgents) {
                int second_index = i;
                if(intentStrategy == intent_strategy::CONSTANT_INTENT_BASED) {
                    second_index = counter;
                }
                boost::random::beta_distribution<> beta_dist(counterfactual_alpha[intent_index][second_index], counterfactual_beta[intent_index][second_index]);
                samples[counter] = beta_dist(gen);
                counter++;
            }
        }
    }
 
    if(intent_index >= 0) {
        // Find the agent with the highest sample value (interventional)
        int best_index = std::distance(samples.begin(), std::max_element(samples.begin(), samples.end()));
        int best_agent = topAgents[best_index];
        (*frequency)[best_agent]++;
        current_index = best_agent;
        counterfactual_index = best_index;
    } else {
        // Use observational choice
        current_index = intent_candidate;
        counterfactual_index = -1;
    }
    return current_index;
}

/**
 * @brief ADDRESS: agent-based normal bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::normal() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> samples(k);

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }

    // Extract the top 10 agents
    std::vector<int> topAgents;
    for (int i = 0; i < k && !queue.empty(); i++) {
        topAgents.push_back(queue.top().second);
        queue.pop();
    }

    // Sample from the Gaussian distribution for each agent
    for (int i = 0 ; i < k ; i++) {
        std::normal_distribution<double> normal_dist(mu[i], std::sqrt(sigma2[i]));
        samples[i] = normal_dist(gen);
    }

    // Find the agent with the minimum sample value
    int best_index = std::distance(samples.begin(), std::min_element(samples.begin(), samples.end()));
    int best_agent = topAgents[best_index];
    (*frequency)[best_agent]++;
    current_index = best_agent;
    return best_agent;
}

/**
 * @brief ADDRESS: location-based normal bandit sampling implementation.
 * 
 * @return int 
 */
int LNS::location_normal() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> samples(regions);

    // Sample from the Gaussian distribution for each agent
    for (int i = 0 ; i < regions ; i++) {
        std::normal_distribution<double> normal_dist(location_mu[i], std::sqrt(location_sigma2[i]));
        samples[i] = normal_dist(gen);
    }

    // Find the agent with the minimum sample value
    int best_index = std::distance(samples.begin(), std::min_element(samples.begin(), samples.end()));
    (*location_frequency)[best_index]++;
    current_index = best_index;
    return best_index;
}

/**
 * @brief ADDRESS: agent-based epsilon greedy bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::topKEpsilonGreedy()
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //build the top k of most delayed
    
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }

    // Extract the top 10 agents
    std::vector<int> topAgents;
    for (int i = 0; i < k && !queue.empty(); i++) {
        topAgents.push_back(queue.top().second);
        queue.pop();
    }
    cout << topAgents.size() << endl;
    //if the first selection or epsilon explore
    double i = dist(rng);
    if (dist(rng) < epsilon) {
        //random arm 
        std::uniform_int_distribution<int> dist2(0, topAgents.size() - 1);
        int t = dist2(rng);
        int rand = topAgents[t];
        (*frequency)[rand]++;
        current_index = rand;
        return rand;
    }
    //best estimate among top k agents
    int min = INT_MAX;
    int index = 0;
    for (auto i : topAgents){
        if ((*q_values)[i] < min || (*frequency)[i] == 0 ){
            index = i;
            min = (*q_values)[i];
        }
    }
    (*frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: agent-based decay epsilon greedy sampeling implementation.
 * 
 * @return int 
 */
int LNS::topKDecayEpsilonGreedy()
{
    std::mt19937 rng(std::random_device{}());
    if (epsilon >= 0){
        epsilon -= decay;
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //build the top k of most delayed
    
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }

    // Extract the top 10 agents
    std::vector<int> topAgents;
    for (int i = 0; i < k && !queue.empty(); i++) {
        topAgents.push_back(queue.top().second);
        queue.pop();
    }
    //if the first selection or epsilon explore
    if (dist(rng) < epsilon) {
        //random arm 
        std::uniform_int_distribution<int> dist2(0, topAgents.size() - 1);
        int rand = topAgents[dist2(rng)];
        (*frequency)[rand]++;
        current_index = rand;
        return rand;
    }
    //best estimate among top k agents
    int min = INT_MAX;
    int index = 0;
    for (auto i : topAgents){
        if ((*q_values)[i] < min || (*frequency)[i] == 0 ){
            index = i;
            min = (*q_values)[i];
        }
    }
    (*frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: location-based UCB-1 bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::location_UCB()
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //build the top k of most delayed
    //first explore everything
    for (int i = 0; i < location_frequency->size(); i++){
        if ((*location_frequency)[i] == 0){
            (*location_frequency)[i]++;
            current_index = i;
            return i;
        }
    }
    
    double min = 0;
    int index = 0;
    for (int i = 0 ; i < location_frequency->size(); i++){
        double sqrt = std::sqrt((2*log(10)) / (*location_frequency)[i]);
        double ucb = (*location_q_values)[i] + (*location_q_values)[i]*sqrt;
        if (ucb < min || min == 0){
            index = i;
            min = ucb;
        }
    }
    (*location_frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: agent-based UCB-1 bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::topKUCB()
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //build the top k of most delayed
    //first explore everything
    for (int i = 0; i < frequency->size(); i++){
        if ((*frequency)[i] == 0){
            (*frequency)[i]++;
            current_index = i;
            return i;
        }
    }
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, AgentComparator> queue;

    // Add all agents to the priority queue
    for (const Agent& agent : agents) {
        int delay = agent.getNumOfDelays();
        int id = agent.id;
        queue.push(std::make_pair(delay, id));
    }

    // Extract the top 10 agents
    std::vector<int> topAgents;
    for (int i = 0; i < 10 && !queue.empty(); i++) {
        topAgents.push_back(queue.top().second);
        queue.pop();
    }

    double min = 0;
    int index = 0;
    for (auto i : topAgents){
        double sqrt = std::sqrt((2*log(10)) / (*frequency)[i]);
        double ucb = (*q_values)[i] + initial_sum_of_costs*sqrt;
        if (ucb < min || min == 0){
            index = i;
            min = ucb;
        }
    }
    (*frequency)[index]++;
    current_index = index;
    return index;
}

/**
 * @brief ADDRESS: agent-based epsilon-greedy bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::decayEpsilonGreedy()
{
    std::mt19937 rng(std::random_device{}());
    cout << "epsilon value : " << epsilon << endl;
    if (epsilon >= 0){
        epsilon -= decay;
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //if the first selection or epsilon explore
        if (dist(rng) < epsilon || std::all_of(frequency->begin(), frequency->end(), [](int i){ return i == 0; })) {
            //random arm 
            std::uniform_int_distribution<int> dist2(0, q_values->size() - 1);
            int rand = dist2(rng);
            (*frequency)[rand]++;
            current_index = rand;
            return rand;
        }
        else{
            //best estimate 
             int min = INT_MAX;
            int index = 0;
            for (int i =0 ; i < q_values->size(); i++){
                if ((*q_values)[i] < min ){
                    index = i;
                    min = (*q_values)[i];
                }
            }
            (*frequency)[index]++;
            current_index = index;
            return index;
        }
}

/**
 * @brief ADDRESS: agent-based epsilon decay bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::epsilonDelay(){
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    //if the first selection or epsilon explore
        if (dist(rng) < epsilon || std::all_of(frequency->begin(), frequency->end(), [](int i){ return i == 0; })) {
            //random arm 
            std::uniform_int_distribution<int> dist(0, q_values->size() - 1);
            int rand = dist(rng);
            (*frequency)[rand]++;
            current_index = rand;
            return rand;
        }
        else{
            //best estimate 
            int index = findMostDelayedAgent();
            (*frequency)[index]++;
            current_index = index;
            return index;
        }
}

/**
 * @brief ADDRESS: agent-based UCB-1 bandit sampeling implementation.
 * 
 * @return int 
 */
int LNS::ucb()
{   
    int max = 0;
    //must explore every agent once 
    for (int i = 0; i < frequency->size(); i++){
        if ((*frequency)[i] == 0){
            (*frequency)[i]++;
            current_index = i;
            return i;
        }
    }
    double min = 0;
    int index = 0;
    for (int i =0 ; i < q_values->size(); i++){
        double sqrt = std::sqrt((2*log(10)) / (*frequency)[i]);
        double ucb = (*q_values)[i] + initial_sum_of_costs * sqrt;
        if (ucb < min || min == 0){
            index = i;
            min = ucb;
        }
    }
    (*frequency)[index]++;
    current_index = index;
    return index;

}

int LNS::findRandomAgent() const
{
    int a = 0;
    int pt = rand() % (sum_of_costs - sum_of_distances) + 1;
    int sum = 0;
    for (; a < (int) agents.size(); a++)
    {
        sum += agents[a].getNumOfDelays();
        if (sum >= pt)
            break;
    }
    assert(sum >= pt);
    return a;
}

// a random walk with path that is shorter than upperbound and has conflicting with neighbor_size agents
void LNS::randomWalk(int agent_id, int start_location, int start_timestep,
                     set<int>& conflicting_agents, int neighbor_size, int upperbound)
{
    int loc = start_location;
    for (int t = start_timestep; t < upperbound; t++)
    {
        auto next_locs = instance.getNeighbors(loc);
        next_locs.push_back(loc);
        while (!next_locs.empty())
        {
            int step = rand() % next_locs.size();
            auto it = next_locs.begin();
            advance(it, step);
            int next_h_val = agents[agent_id].path_planner->my_heuristic[*it];
            if (t + 1 + next_h_val < upperbound) // move to this location
            {
                path_table.getConflictingAgents(agent_id, conflicting_agents, loc, *it, t + 1);
                loc = *it;
                break;
            }
            next_locs.erase(it);
        }
        if (next_locs.empty() || conflicting_agents.size() >= neighbor_size)
            break;
    }
}

void LNS::validateSolution() const
{
    int sum = 0;
    for (const auto& a1_ : agents)
    {
        if (a1_.path.empty())
        {
            cerr << "No solution for agent " << a1_.id << endl;
            exit(-1);
        }
        else if (a1_.path_planner->start_location != a1_.path.front().location)
        {
            cerr << "The path of agent " << a1_.id << " starts from location " << a1_.path.front().location
                << ", which is different from its start location " << a1_.path_planner->start_location << endl;
            exit(-1);
        }
        else if (a1_.path_planner->goal_location != a1_.path.back().location)
        {
            cerr << "The path of agent " << a1_.id << " ends at location " << a1_.path.back().location
                 << ", which is different from its goal location " << a1_.path_planner->goal_location << endl;
            exit(-1);
        }
        for (int t = 1; t < (int) a1_.path.size(); t++ )
        {
            if (!instance.validMove(a1_.path[t - 1].location, a1_.path[t].location))
            {
                cerr << "The path of agent " << a1_.id << " jump from "
                     << a1_.path[t - 1].location << " to " << a1_.path[t].location
                     << " between timesteps " << t - 1 << " and " << t << endl;
                exit(-1);
            }
        }
        sum += (int) a1_.path.size() - 1;
        for (const auto  & a2_: agents)
        {
            if (a1_.id >= a2_.id || a2_.path.empty())
                continue;
            const auto & a1 = a1_.path.size() <= a2_.path.size()? a1_ : a2_;
            const auto & a2 = a1_.path.size() <= a2_.path.size()? a2_ : a1_;
            int t = 1;
            for (; t < (int) a1.path.size(); t++)
            {
                if (a1.path[t].location == a2.path[t].location) // vertex conflict
                {
                    cerr << "Find a vertex conflict between agents " << a1.id << " and " << a2.id <<
                            " at location " << a1.path[t].location << " at timestep " << t << endl;
                    exit(-1);
                }
                else if (a1.path[t].location == a2.path[t - 1].location &&
                        a1.path[t - 1].location == a2.path[t].location) // edge conflict
                {
                    cerr << "Find an edge conflict between agents " << a1.id << " and " << a2.id <<
                         " at edge (" << a1.path[t - 1].location << "," << a1.path[t].location <<
                         ") at timestep " << t << endl;
                    exit(-1);
                }
            }
            int target = a1.path.back().location;
            for (; t < (int) a2.path.size(); t++)
            {
                if (a2.path[t].location == target)  // target conflict
                {
                    cerr << "Find a target conflict where agent " << a2.id << " (of length " << a2.path.size() - 1<<
                         ") traverses agent " << a1.id << " (of length " << a1.path.size() - 1<<
                         ")'s target location " << target << " at timestep " << t << endl;
                    exit(-1);
                }
            }
        }
    }
    if (sum_of_costs != sum)
    {
        cerr << "The computed sum of costs " << sum_of_costs <<
             " is different from the sum of the paths in the solution " << sum << endl;
        exit(-1);
    }
}

void LNS::writeIterStatsToFile(const string & file_name) const
{
    if (init_lns != nullptr)
    {
        init_lns->writeIterStatsToFile(file_name + "-initLNS.csv");
    }
    if (iteration_stats.size() <= 1)
        return;
    string name = file_name;
    if (use_init_lns or num_of_iterations > 0)
        name += "-LNS.csv";
    else
        name += "-" + init_algo_name + ".csv";
    std::ofstream output;
    output.open(name);
    // header
    cout << "======"<< ALNS << "=======" << endl;
    if (1 > 0){
        output << "num of agents," <<
            "sum of costs," <<
            "weights," << 
            "runtime," <<
            "cost lowerbound," <<
            "sum of distances," <<
            "MAPF algorithm" << endl;
            for (const auto &data : iteration_stats)
            {
                output << data.num_of_agents << "," <<
                    data.sum_of_costs << "," << data.weights << "," << data.runtime << "," <<
                    max(sum_of_costs_lowerbound, sum_of_distances) << "," <<
                    sum_of_distances << "," <<
                    data.algorithm << endl;
            }
    }
    else {
        output << "num of agents," <<
            "sum of costs," <<
            "runtime," <<
            "cost lowerbound," <<
            "sum of distances," <<
            "MAPF algorithm" << endl;
        for (const auto &data : iteration_stats)
        {
            output << data.num_of_agents << "," <<
                data.sum_of_costs << "," <<
                data.runtime << "," <<
                max(sum_of_costs_lowerbound, sum_of_distances) << "," <<
                sum_of_distances << "," <<
                data.algorithm << endl;
        }
    }
    
    output.close();
}

void LNS::writeResultToFile(const string & file_name) const
{
    if (init_lns != nullptr)
    {
        init_lns->writeResultToFile(file_name + "-initLNS.csv", sum_of_distances, preprocessing_time);
    }
    string name = file_name;
    if (use_init_lns or num_of_iterations > 0)
        name += "-LNS.csv";
    else
        name += "-" + init_algo_name + ".csv";
    std::ifstream infile(name);
    bool exist = infile.good();
    infile.close();
    if (!exist)
    {
        ofstream addHeads(name);
        addHeads << "runtime,solution cost,initial solution cost,lower bound,sum of distance," <<
                 "iterations," <<
                 "group size," <<
                 "runtime of initial solution,restart times,area under curve," <<
                 "LL expanded nodes,LL generated,LL reopened,LL runs," <<
                 "preprocessing runtime,solver name,instance name,success,selected_neighbor,neighbor_size" << endl;
        addHeads.close();
    }
    uint64_t num_LL_expanded = 0, num_LL_generated = 0, num_LL_reopened = 0, num_LL_runs = 0;
    for (auto & agent : agents)
    {
        agent.path_planner->reset();
        num_LL_expanded += agent.path_planner->accumulated_num_expanded;
        num_LL_generated += agent.path_planner->accumulated_num_generated;
        num_LL_reopened += agent.path_planner->accumulated_num_reopened;
        num_LL_runs += agent.path_planner->num_runs;
    }
    double auc = 0;
    if (!iteration_stats.empty())
    {
        auto prev = iteration_stats.begin();
        auto curr = prev;
        ++curr;
        while (curr != iteration_stats.end() && curr->runtime < time_limit)
        {
            auc += (prev->sum_of_costs - sum_of_distances) * (curr->runtime - prev->runtime);
            prev = curr;
            ++curr;
        }
        auc += (prev->sum_of_costs - sum_of_distances) * (time_limit - prev->runtime);
    }
    ofstream stats(name, std::ios::app);
    stats << runtime << "," << sum_of_costs << "," << initial_sum_of_costs << "," <<
          sum_of_costs_lowerbound << "," << sum_of_distances << "," <<
          iteration_stats.size() << "," << average_group_size << "," <<
          initial_solution_runtime << "," << restart_times << "," << auc << "," <<
          num_LL_expanded << "," << num_LL_generated << "," << num_LL_reopened << "," << num_LL_runs << "," <<
          preprocessing_time << "," << getSolverName() << "," << instance.getInstanceName() << "," << iteration_stats.back().success << "," << selected_neighbor << "," << neighbor_size << endl;
    stats.close();
}

void LNS::writePathsToFile(const string & file_name) const
{
    std::ofstream output;
    output.open(file_name);

    for (const auto &agent : agents)
    {
        output << "Agent " << agent.id << ":";
        for (const auto &state : agent.path)
            output << "(" << instance.getRowCoordinate(state.location) << "," <<
                            instance.getColCoordinate(state.location) << ")->";
        output << endl;
    }
    output.close();
}
