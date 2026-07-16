#include "LNS.h"
#include "ECBS.h"
#include <queue>
#include <cmath>     // std::log/std::exp for the SPSA softplus acceptance policy (explicit for Linux/gcc build)
#include <cstdlib>   // std::getenv/atof/atoi/rand used by the AMOR_ACCEPT acceptance modes
#include <array>
#include <algorithm>
#include <random>

// ===== CACo selective-EECBS instrumentation (file-static: one LNS run per process) =====
static uint64_t total_eecbs_calls = 0;
static uint64_t total_eecbs_hl_expanded = 0;
static FILE* g_ord_probe_fp = nullptr;   // AMOR_ORDER_PROBE counterfactual repair-order logger
static bool  g_ord_probe_open = false;

// ===== Learned / bandit destroy seed-selection state (file-static: one LNS run per process) =====
namespace {
  // mode 10 = SMOOTH feature policy over the top-K most-delayed (bounded downside, like ADDRESS/TACKLE):
  // softmax(theta.phi)/tau, Gumbel-max, contextual-bandit REINFORCE + L2 pull to warm-start (do-no-harm).
  const  double g_sd_W0[4] = {3.0, 0.0, 0.0, 0.0};   // warm-start prior: pure high-delay (hard tabu gives diversity) == stock argmax+tabu
  double g_sd_theta[4] = {3.0, 0.0, 0.0, 0.0};        // [log-delay, relative-delay, staleness, realized-improvement EMA]
  double g_sd_m[4]={0,0,0,0}, g_sd_v[4]={0,0,0,0}; int g_sd_t=0;
  double g_sd_phibar[4]={0,0,0,0}, g_sd_phi_chosen[4]={0,0,0,0}, g_sd_baseline=0.0;
  int    g_sd_last=-1; long g_sd_iter=0; bool g_sd_cfg=false; double g_sd_r2=0.0;
  double g_sd_tau=0.15, g_sd_alpha=0.05, g_sd_lam=0.002; int g_sd_K=32; double g_sd_beta=0.0;   // beta = de-confound weight on the seed's OWN delay reduction (0=off default; wash locally, ablation knob for scale)
  std::vector<double> g_sd_yema; std::vector<int> g_sd_touched;
  // mode 20 = ADDRESS-style per-agent Thompson bandit over top-K most-delayed (Gaussian-approx Beta posterior).
  std::vector<double> g_ts_a, g_ts_b; int g_ts_last=-1, g_ts_K=32; bool g_ts_cfg=false;
  double sd_gauss(){ double u1=((double)rand()+1.0)/((double)RAND_MAX+2.0), u2=(double)rand()/(double)RAND_MAX;
                     return std::sqrt(-2.0*std::log(u1))*std::cos(6.283185307179586*u2); }
  void sd_cfg(){ if(g_sd_cfg) return; g_sd_cfg=true; const char* s;
    if((s=std::getenv("AMOR_SD_TAU")))   g_sd_tau=atof(s);
    if((s=std::getenv("AMOR_SD_ALPHA"))) g_sd_alpha=atof(s);
    if((s=std::getenv("AMOR_SD_LAM")))   g_sd_lam=atof(s);
    if((s=std::getenv("AMOR_SD_K")))     g_sd_K=std::max(1,atoi(s));
    if((s=std::getenv("AMOR_SD_BETA")))  g_sd_beta=atof(s); }
  void ts_cfg(){ if(g_ts_cfg) return; g_ts_cfg=true; const char* s;
    if((s=std::getenv("AMOR_TS_K"))) g_ts_K=std::max(1,atoi(s)); }
  // mode 30 = TACKLE MABUC (counterfactual K x K Beta table + Thompson, roulette intent) — matched in-binary baseline.
  std::vector<std::vector<int>> g_tk_alpha, g_tk_beta; int g_tk_intent=-1, g_tk_col=-1;
  double sample_beta(int a, int b){
    // seeded once from the solver's srand(--seed) stream so each run seed gets its OWN Thompson
    // exploration realization (audit: fixed 12345 made 25-seed capture CIs treat TS randomness as
    // a constant); drawing from rand() here leaves the stock random_shuffle stream untouched
    // because this runs only on TS pulls, which never occur on the byte-identical locked path.
    static thread_local std::mt19937 g((unsigned)rand());
    std::gamma_distribution<double> ga((double)a,1.0), gb((double)b,1.0);
    double x=ga(g), y=gb(g); return (x+y>0)?x/(x+y):0.5; }
  // learner-private uniform for V5 randomized rounding (audit F2 / plan FF-4). v2: seeded from a hash of
  // the first reward's trajectory values — ZERO stock rand() consumption ever (v1 drew one rand() at init,
  // which shifted the solver stream at pull 1 and broke whole-run byte-identity on LOCKed runs). Same run
  // seed -> same init trajectory -> same hash -> reproducible; different seeds decouple naturally.
  double amor_u01(unsigned ent){ static thread_local std::mt19937 g(ent*2654435761u ^ 0x9e3779b9u);
    return std::uniform_real_distribution<double>(0.0,1.0)(g); }
  // REPAIR-ORDER = Plackett-Luce learned PP priority: sample the order ~ softmax(theta.phi)/tau without replacement.
  // theta=0 => flat scores => uniform-random permutation == stock random_shuffle (do-no-harm). REINFORCE, reward=-slack.
  double g_ro_theta[2] = {0.0, 0.0};   // features: [z(free-flow dist), z(delay)]
  double g_ro_m[2]={0,0}, g_ro_v[2]={0,0}; int g_ro_t=0;
  double g_ro_grad[2]={0,0}, g_ro_baseline=0.0, g_ro_lb=0.0, g_ro_r2=0.0;
  bool   g_ro_have=false, g_ro_cfg=false; double g_ro_tau=1.0, g_ro_alpha=0.02;
  void ro_cfg(){ if(g_ro_cfg) return; g_ro_cfg=true; const char* s;
    if((s=std::getenv("AMOR_RO_TAU")))   g_ro_tau=atof(s);
    if((s=std::getenv("AMOR_RO_ALPHA"))) g_ro_alpha=atof(s); }
  // REPAIR-BANDIT (AMOR_REPAIR=20): epsilon-greedy-EMA over 5 fixed PP-order rules — the discrete comparator to the
  // smooth Plackett-Luce learner (mirrors accept mode-8). Arms: 0=random(stock) 1=longest-haul 2=shortest 3=most-delayed 4=least-delayed.
  double g_rb_val[5]={0,0,0,0,0}; int g_rb_n[5]={0,0,0,0,0}; int g_rb_last=-1; double g_rb_lb=0.0;
  double g_rb_eps=0.15, g_rb_alpha=0.2; bool g_rb_cfg=false;
  bool g_rb_allow[5]={true,true,true,true,true};   // AMOR_RB_ARMS="0,1,3" restricts the arm set (drop dominated arms)
  void rb_cfg(){ if(g_rb_cfg) return; g_rb_cfg=true; const char* s;
    if((s=std::getenv("AMOR_RB_EPS")))   g_rb_eps=atof(s);
    if((s=std::getenv("AMOR_RB_ALPHA"))) g_rb_alpha=atof(s);
    if((s=std::getenv("AMOR_RB_ARMS"))){ for(int a=0;a<5;a++) g_rb_allow[a]=false;
      std::string v=s; size_t p=0; while(p<v.size()){ size_t q=v.find(',',p); if(q==std::string::npos)q=v.size();
        int a=atoi(v.substr(p,q-p).c_str()); if(a>=0&&a<5) g_rb_allow[a]=true; p=q+1; } } }
  // AMOR-CUCB (AMOR_REPAIR=21): CONSERVATIVE UCB order-bandit — the paper's learning element (pre-registered
  // in PREREGISTRATION_cucb.md). Slot 0 = ANCHOR arm (random_shuffle = stock LNS2). Plays argmax-UCB but a
  // SAFETY GATE forces the anchor unless the realized-reward ledger certifies (1-alpha) of the anchor's
  // cumulative reward is preserved w.h.p. -> anytime do-no-harm floor (Wu ICML'16 pattern, unknown baseline).
  // No free warmup pulls: unexplored arms have UCB=+inf but enter ONLY through the gate.
  double g_cu_sum[5]={0,0,0,0,0}, g_cu_sq[5]={0,0,0,0,0}; int g_cu_n[5]={0,0,0,0,0}; long g_cu_t=0; double g_cu_S=0.0;
  int g_cu_last=-1; double g_cu_lb=0.0; long g_cu_gate_open=-1;
  int g_cu_arms[5]={0,1,4,-1,-1}; int g_cu_K=3;          // slot->arm-id map; arm ids as mode 20 (0=rand 1=long 2=short 3=mdel 4=ldel)
  double g_cu_alpha=0.25, g_cu_delta=0.05; int g_cu_reward=0; bool g_cu_cfg=false;   // reward 0=completion-indicator 1=slack-fraction
  int g_cu_mode=0; bool g_cu_locked=false;   // PLAN B (pre-registered): mode 1 = UCB1 proposal + one-way circuit BREAKER
  // BG-SE (mode 2, AMOR_CU_MODE=budget, PRE-REGISTERED V2): budget-gated successive elimination.
  bool g_cu_gated=false, g_cu_lockbg=false, g_cu_elim[5]={false,false,false,false,false};
  int g_cu_rr=0; double g_cu_That=0, g_cu_Nid=0;
  // V4 (pre-reg 2026-07-03): confidence-gated departure (AMOR_CU_MODE=cts). Pre-departure the anchor
  // plays except a beta challenger lane; departure only when a challenger's Beta-posterior mean beats
  // the anchor's at one-sided delta=0.05 (z>1.645) with n_j>=25; one-way; post-departure = exact V3.
  bool g_cu_conf=false, g_cu_dep=false; long g_cu_dep_iter=-1;
  int    g_cu_warm=100; double g_cu_dmin=0.15, g_cu_gam=0.30, g_cu_beta=0.20;   // Dmin default == frozen V3 spec (audit: 0.10 inflated Nid 2.25x when env unset)
  // mode 4 = Ropke-Pisinger'06 roulette-wheel baseline (ALNS necessity-table row; NO safety machinery):
  // segment=100 pulls, reaction w=0.8w+0.2*(score/pulls); adapted scores for greedy LNS: 33 improving
  // repair / 13 completed-not-improving / 0 failed (RP's sigma3 "accepted-worse" cannot occur under greedy accept).
  double g_cu_w[5]={1,1,1,1,1}, g_cu_sc[5]={0,0,0,0,0}; int g_cu_scn[5]={0,0,0,0,0};
  void cu_cfg(){ if(g_cu_cfg) return; g_cu_cfg=true; const char* s;
    if((s=std::getenv("AMOR_RB_ALPHA_BUDGET"))) g_cu_alpha=atof(s);
    if((s=std::getenv("AMOR_RB_DELTA")))        g_cu_delta=atof(s);
    if((s=std::getenv("AMOR_RB_REWARD")))       g_cu_reward=(std::string(s)=="slackfrac")?1:((std::string(s)=="rawdelta")?2:0);   // V5: rawdelta = randomized rounding of (old-new)/16 -> valid Bernoulli, magnitude-aware (Phase-0-validated ranking)
    if((s=std::getenv("AMOR_CU_MODE"))){ std::string v=s;
      g_cu_mode=(v=="breaker")?1:((v=="budget")?2:((v=="ts")?3:((v=="roulette")?4:0)));
      if(v=="cts"){ g_cu_mode=3; g_cu_conf=true; }
      if(v=="etc"){ g_cu_mode=6; } }   // ETC-50 necessity baseline (identify-then-commit; plan FF-10)
    if((s=std::getenv("AMOR_CU_WARMUP")))       g_cu_warm=std::max(1,atoi(s));   // audit #5: guard div-by-zero
    if((s=std::getenv("AMOR_CU_DMIN")))         g_cu_dmin=atof(s);
    if((s=std::getenv("AMOR_CU_GAMMA")))        g_cu_gam=atof(s);
    if((s=std::getenv("AMOR_CU_BETA")))         g_cu_beta=atof(s);
    if((s=std::getenv("AMOR_CU_ARMS"))){ g_cu_K=0; std::string v=s; size_t p=0;
      while(p<v.size() && g_cu_K<5){ size_t q=v.find(',',p); if(q==std::string::npos)q=v.size();
        int aid=atoi(v.substr(p,q-p).c_str()); if(aid<0)aid=0; if(aid>4)aid=4;   // U0: clamp arm ids
        g_cu_arms[g_cu_K++]=aid; p=q+1; } if(g_cu_K==0){g_cu_arms[0]=0;g_cu_K=1;}
      if(g_cu_arms[0]!=0){ g_cu_arms[0]=0; }   // U0: slot 0 MUST be the anchor (gate/breaker math assumes it)
    }
    if(g_cu_mode==3 && g_cu_reward==1){ g_cu_reward=0;   // ts requires Bernoulli rewards: fractional slackfrac invalid; rawdelta (2) IS Bernoulli via randomized rounding
      std::cerr << "AMOR: mode=ts forces completion reward (slackfrac ignored)" << std::endl; } }
  // PROBE (AMOR_SEED=39): uniform-random top-K delayed seed (unbiased) + per-iter feature/reward log to AMOR_LOG.
  // Make-or-break residual-signal test for the contextual-bandit rebuild — NO bandit machinery, just printf.
  int g_pr_last=-1; long g_pr_iter=0; double g_pr_phi[6]={0,0,0,0,0,0}; std::vector<int> g_pr_touched;
  FILE* g_pr_fp=nullptr; bool g_pr_open=false;
  // ===== CONTEXTUAL LINEAR THOMPSON bandit for the seed (AMOR_SEED=40=LinTS, 41=LinUCB). d=6 strictly-O(1) features:
  // [bias, log-delay, rel-delay, path-stretch, staleness, 1/degree]. Thompson exploration (uncertainty-driven, auto-
  // scales at data-sparse/large N) + argmax(delay) prior (do-no-harm; AMOR_LB_V=0 => pure argmax). Recovers argmax /
  // TACKLE / ADDRESS as special cases; the cross-scale no-regret vehicle.
  const int LTD=6;
  double g_lt_Ainv[6][6]={{0}}, g_lt_b[6]={0}, g_lt_theta[6]={0}, g_lt_xlast[6]={0};
  double g_lt_mu[6]={0}, g_lt_var[6]={1,1,1,1,1,1}, g_lt_rmean=0, g_lt_r2=0;
  int g_lt_last=-1; long g_lt_iter=0; std::vector<int> g_lt_touched; bool g_lt_init=false, g_lt_cfg=false;
  double g_lt_v=1.0, g_lt_lam=1.0, g_lt_lamD=50.0, g_lt_mud=3.0, g_lt_alpha=1.0; int g_lt_K=32;
  void lt_cfg(){ if(g_lt_cfg)return; g_lt_cfg=true; const char* s;
    if((s=std::getenv("AMOR_LB_V")))    g_lt_v=atof(s);
    if((s=std::getenv("AMOR_LB_LAMD"))) g_lt_lamD=atof(s);
    if((s=std::getenv("AMOR_LB_MUD")))  g_lt_mud=atof(s);
    if((s=std::getenv("AMOR_LB_ALPHA")))g_lt_alpha=atof(s);
    if((s=std::getenv("AMOR_TS_K")))    g_lt_K=std::max(2,atoi(s)); }
  void lt_setup(){ if(g_lt_init)return; g_lt_init=true;
    for(int i=0;i<LTD;i++){ for(int j=0;j<LTD;j++) g_lt_Ainv[i][j]=0.0;
      g_lt_Ainv[i][i]=1.0/((i==1)?g_lt_lamD:g_lt_lam); }   // A0=diag(lam..lamD@delay..); Ainv=inverse
    for(int i=0;i<LTD;i++) g_lt_b[i]=0.0; g_lt_b[1]=g_lt_lamD*g_lt_mud;   // b=A0*mu0, mu0=[0,mud,0..]
    for(int i=0;i<LTD;i++) g_lt_theta[i]=g_lt_b[i]*g_lt_Ainv[i][i]; }     // theta=Ainv*b=mu0
}

LNS::LNS(const Instance& instance, double time_limit, const string & init_algo_name, const string & replan_algo_name,
         const string & destory_name, int neighbor_size, int num_of_iterations, bool use_init_lns,
         const string & init_destory_name, bool use_sipp, int screen, PIBTPPS_option pipp_option) :
         BasicLNS(instance, time_limit, neighbor_size, screen),
         init_algo_name(init_algo_name),  replan_algo_name(replan_algo_name), num_of_iterations(num_of_iterations),
         use_init_lns(use_init_lns),init_destory_name(init_destory_name),
         path_table(instance.map_size), pipp_option(pipp_option)
{
    start_time = Time::now();
    replan_time_limit = time_limit / 100;
    if (destory_name == "Adaptive")
    {
        ALNS = true;
        destroy_weights.assign(DESTORY_COUNT, 1);
        decay_factor = 0.01;
        reaction_factor = 0.01;
    }
    else if (destory_name == "RandomWalk")
        destroy_strategy = RANDOMWALK;
    else if (destory_name == "Intersection")
        destroy_strategy = INTERSECTION;
    else if (destory_name == "Random")
        destroy_strategy = RANDOMAGENTS;
    else
    {
        cerr << "Destroy heuristic " << destory_name << " does not exists. " << endl;
        exit(-1);
    }

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
                    replan_algo_name,init_destory_name, neighbor_size, screen);
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

    iteration_stats.emplace_back(neighbor.agents.size(),
                                 initial_sum_of_costs, initial_solution_runtime, init_algo_name);
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

    // ── Zero-overhead non-greedy acceptance state (AMOR_ACCEPT) ──────────────────
    // LAHC keeps a length-L history of past global costs; record-to-record uses best+delta.
    // All O(1) per iteration: NO extra search, so iteration throughput is identical to greedy LNS2.
    static const int _accept_init = [](){ const char* s=std::getenv("AMOR_ACCEPT");
        if(!s) return 0; std::string v=s; if(v=="lahc")return 1; if(v=="rr")return 2; if(v=="reheat")return 3; if(v=="ta")return 4; if(v=="adapt")return 5; if(v=="adaptc")return 6; if(v=="adaptsw")return 7; if(v=="bandit")return 8; if(v=="spsa")return 9; if(v=="pg")return 10; if(v=="gd")return 11; return 0; }();
    static const int _lahc_L = [](){ const char* s=std::getenv("AMOR_LAHC_L"); return s?std::max(1,atoi(s)):50; }();
    static const double _rr_delta = [](){ const char* s=std::getenv("AMOR_RR_DELTA"); return s?atof(s):0.0; }();
    // AMOR_SWITCH_FRAC f in [0,1]: ORACLE time-switch for the within-run non-stationarity test.
    // Apply the AMOR_ACCEPT criterion for the first f fraction of wall-clock, then force GREEDY.
    // f=0 => greedy from the start (== pure greedy); f=1 (default -1 = disabled) => never switch (pure criterion).
    // A schedule (interior f) beating BOTH endpoints proves real within-run signal (bandit has something to learn).
    static const double _switch_frac = [](){ const char* s=std::getenv("AMOR_SWITCH_FRAC"); return s?atof(s):-1.0; }();
    // AMOR_ACCEPT=gd great-deluge: accept iff cand <= best + d0*(1-tau), tau=elapsed/T (budget-tied,
    // N-adaptive without iteration counts). Early exploratory like rr; late (tau->1) d0->0 == greedy-on-best.
    static const double _gd_d0 = [](){ const char* s=std::getenv("AMOR_GD_D0"); return s?atof(s):20.0; }();
    // AMOR_BEST_RETURN: return the BEST INCUMBENT visited, not the last working solution. The stock harness
    // (and every destroy-bandit paper) returns the last working solution — correct under greedy, but it
    // MISMEASURES non-greedy accept (which can end on an uphill-accepted worse solution). Togglable so we can
    // ablate rr-with vs rr-without best-return (attribution). Default OFF = byte-for-byte stock behaviour.
    static const bool _best_return = (std::getenv("AMOR_BEST_RETURN") != nullptr);
    // Adaptive-delta (AMOR_ACCEPT=adapt|adaptc): slack grows with stagnation, delta_eff = min(cap, slope*iters_since_improve).
    // adapt = best-reference (rr-style base, Candidate 1); adaptc = current-reference (pure ta-adaptive, an extra arm).
    // Candidate 2 is adaptsw (reference-switching, below). Auto-greedy while improving (iters_since_improve=0 =>
    // delta=0), auto-explore when deeply stuck. The absolute
    // stagnation count self-scales to the iteration budget => the same rule is greedy on big/slow maps (few iters,
    // count never accumulates) and rr-like on small/fast maps when stuck (count grows). One mechanism, both regimes.
    // delta_eff = min(cap, floor + slope*iters_since_improve). The FLOOR is a constant always-on slack: it
    // generalizes the family so one formula subsumes all criteria — floor=slope=0 => greedy; floor=delta,slope=0
    // => record-to-record (constant); floor=0,slope>0 => pure stagnation-adaptive; floor>0,slope>0 => rr + extra
    // escape when stuck. Needed because on high-gap maps improvements reset the stagnation count constantly, so
    // pure-adaptive (floor=0) collapses to greedy and misses rr's early-exploration win; a small floor restores it.
    static const double _adapt_floor = [](){ const char* s=std::getenv("AMOR_ADAPT_FLOOR"); return s?atof(s):0.0; }();
    static const double _adapt_slope = [](){ const char* s=std::getenv("AMOR_ADAPT_SLOPE"); return s?atof(s):0.5; }();
    static const double _adapt_cap   = [](){ const char* s=std::getenv("AMOR_ADAPT_CAP");   return s?atof(s):20.0; }();
    // Candidate 2 (AMOR_ACCEPT=adaptsw): reference SWITCHES on stagnation depth. While shallowly stuck
    // (iters_since_improve < SWITCH) anchor to the CURRENT plan (threshold-accepting => bold relocation, the
    // mid-phase win); once deeply stuck (>= SWITCH) anchor to the BEST plan (record-to-record => late refine).
    // Captures ta's mid advantage AND rr's late advantage in one criterion. Same delta_eff schedule as adapt.
    static const int    _adapt_switch= [](){ const char* s=std::getenv("AMOR_ADAPT_SWITCH"); return s?std::max(0,atoi(s)):30; }();
    // ── Multi-armed bandit over acceptance criteria (AMOR_ACCEPT=bandit) ─────────────────────────────────
    // Arms = fixed (reference, delta) acceptance policies spanning exploit->explore. Every W iterations the
    // bandit observes reward = best-SOC improvement during that window and reselects an arm (epsilon-greedy
    // with recency-weighted value -> handles NON-STATIONARITY, since the best criterion shifts by search phase
    // and map). CRUCIAL: reward is best-so-far improvement (NOT current-SOC change) — a current-SOC reward
    // would punish every uphill move and collapse the bandit to greedy. Selection/update are O(1) (zero-overhead).
    static const int    _bandit_W     = [](){ const char* s=std::getenv("AMOR_BANDIT_W");     return s?std::max(1,atoi(s)):64; }();
    static const double _bandit_eps   = [](){ const char* s=std::getenv("AMOR_BANDIT_EPS");   return s?atof(s):0.15; }();
    static const double _bandit_alpha = [](){ const char* s=std::getenv("AMOR_BANDIT_ALPHA"); return s?atof(s):0.2; }();
    // SPSA-tuned continuous acceptance policy (AMOR_ACCEPT=spsa): delta_eff = softplus(theta . [1, s, u]),
    // s = normalized stagnation, u = time progress. theta is tuned ONLINE per-run by sign-SPSA on a windowed
    // best-improvement reward (no labels, no offline training; each instance learns its own delta schedule,
    // so it is scale-free across maps by construction). theta is initialized to give delta_eff ~ 5 (rr-like),
    // a safe floor: worst case it stays ~rr, best case SPSA finds a better per-map schedule.
    static const int    _spsa_W    = [](){ const char* s=std::getenv("AMOR_SPSA_W");     return s?std::max(1,atoi(s)):64; }();
    static const double _spsa_a    = [](){ const char* s=std::getenv("AMOR_SPSA_A");     return s?atof(s):0.05; }(); // small step: avoids random-walk drift off the good init
    static const double _spsa_c    = [](){ const char* s=std::getenv("AMOR_SPSA_C");     return s?atof(s):0.6; }();
    static const double _spsa_snorm= [](){ const char* s=std::getenv("AMOR_SPSA_SNORM"); return s?atof(s):50.0; }();
    static const double _spsa_dmax = [](){ const char* s=std::getenv("AMOR_SPSA_DMAX");  return s?atof(s):50.0; }();
    static const int    _spsa_dim  = [](){ const char* s=std::getenv("AMOR_SPSA_DIM");   int d=s?atoi(s):5; return (d<=3)?3:5; }(); // 3=tolerance only (best-ref), 5=+anchor blend
    // F1 (AMOR_SPSA_DINIT): warm-start SPSA theta0 to the rr anchor delta. Default 0.0 = cold start (== current).
    static const double _spsa_dinit= [](){ const char* s=std::getenv("AMOR_SPSA_DINIT"); return s?atof(s):0.0; }();
    // F9 (AMOR_SPSA_SCHED): annealed a_k/c_k + freeze-best-theta. Default 0 = constant step (== current).
    static const int    _spsa_sched= [](){ const char* s=std::getenv("AMOR_SPSA_SCHED"); return s?atoi(s):0; }();
    static const double _spsa_a0   = [](){ const char* s=std::getenv("AMOR_SPSA_A0");    return s?atof(s):0.30; }();
    static const double _spsa_at   = [](){ const char* s=std::getenv("AMOR_SPSA_AT");    return s?atof(s):0.02; }();
    static const double _spsa_ct   = [](){ const char* s=std::getenv("AMOR_SPSA_CT");    return s?atof(s):0.15; }();
    static const double _spsa_frz  = [](){ const char* s=std::getenv("AMOR_SPSA_FRZ");   return s?atof(s):0.34; }();
    // Policy-gradient acceptance (AMOR_ACCEPT=pg): the SAME 5-param policy, but learned by REINFORCE instead of
    // SPSA. Acceptance of UPHILL moves is made stochastic & differentiable: lambda=sigmoid(t3+t4*s),
    // delta=softplus(t0+t1*s+t2*u), threshold T=best+lambda*(cur-best)+delta, accept ~ Bernoulli(sigmoid((T-cand)/tau)).
    // Per window: reward R=relative best-drop; advantage A=R-baseline; theta += Adam( A * sum_decisions grad_log_pi ).
    // Uses EVERY uphill decision's analytic gradient (forward collects grad_log_pi, backward weights by reward) ->
    // far more information per update than SPSA's 1 scalar -> learns much faster. Improvements (cand<=cur) always accepted.
    static const double _pg_tau = [](){ const char* s=std::getenv("AMOR_PG_TAU"); return s?atof(s):5.0; }();   // softness temperature
    static const double _pg_lr  = [](){ const char* s=std::getenv("AMOR_PG_LR");  return s?atof(s):0.02; }();   // Adam learning rate
    static const int    _pg_W   = [](){ const char* s=std::getenv("AMOR_PG_W");   if(s) return std::max(1,atoi(s));
        const char* a=std::getenv("AMOR_PG_ANTITHETIC"); return (a && atoi(a))?32:64; }(); // F8: halve default window when antithetic (preserve update freq)
    static const double _pg_dinit = [](){ const char* s=std::getenv("AMOR_PG_DINIT"); return s?atof(s):2.0; }();  // FIX1: warm-start tolerance delta (rr2-like, best-anchored); 0 = old greedy init
    static const double _pg_gamma = [](){ const char* s=std::getenv("AMOR_PG_GAMMA"); return s?atof(s):0.98; }(); // FIX2: eligibility-trace decay (credit horizon ~ 1/(1-gamma) iters)
    // F4 (AMOR_PG_KNORM): delta units. 0=absolute(current), 1=best/N(legacy FIX4), 2=per-agent optimality gap.
    // Default: 0 unless legacy AMOR_PG_KAPPA is set (then 1). AMOR_PG_DFRAC = warm-start delta as a FRACTION of the normalizer.
    static const int    _pg_knorm = [](){ const char* s=std::getenv("AMOR_PG_KNORM"); if(s) return atoi(s); return std::getenv("AMOR_PG_KAPPA")?1:0; }();
    static const double _pg_dfrac = [](){ const char* s=std::getenv("AMOR_PG_DFRAC"); return s?atof(s):0.25; }();
    // FIX5 = PGPE (parameter-exploration policy gradient): perturb the policy PARAMS per window, act DETERMINISTICALLY
    // within the window, reward = windowed best-drop, gradient = (R-baseline)*eps/sigma^2. Removes per-iteration
    // stochastic-acceptance noise (the soft policy is a noisy rr) AND matches reward/action granularity (like the
    // bandit, but continuous). _pg_pgpe=1 (default) uses PGPE; =0 falls back to soft-acceptance REINFORCE (ablation).
    static const int    _pg_pgpe  = [](){ const char* s=std::getenv("AMOR_PG_PGPE");  return s?atoi(s):1; }();
    static const double _pg_sigma = [](){ const char* s=std::getenv("AMOR_PG_SIGMA"); return s?atof(s):0.5; }();   // PGPE param-exploration stddev
    static const double _pg_t1init= [](){ const char* s=std::getenv("AMOR_PG_T1INIT"); return s?atof(s):0.0; }();  // FIX6: warm-start theta1 (stagnation slope) -> start as an ADAPTIVE (open-when-stuck) schedule, matching the bandit's mechanism
    // FIX7: iteration-budget regime feature. delta opens (theta0 active) only once enough iterations have accrued.
    // r = min(1, iters_done/IREF): r~0 early or on congested maps (few iters) -> delta~softplus(0)~greedy (protects the
    // tail where fixed rr HURTS); r~1 on iteration-rich mid maps -> delta=softplus(theta0)~rr5 (explores where it wins).
    // This is the continuous analogue of the bandit's greedy-early/explore-late switching. IREF=0 -> r=1 (flat delta).
    static const double _pg_iref  = [](){ const char* s=std::getenv("AMOR_PG_IREF"); return s?atof(s):3000.0; }();
    // F5 (AMOR_PG_IREFMODE): iteration-richness gate mode. 1=legacy fixed iref (current/DEFAULT), 0=flat r=1, 2=online iref.
    static const int    _pg_irefmode = [](){ const char* s=std::getenv("AMOR_PG_IREFMODE"); return s?atoi(s):1; }();
    static const double _pg_irefc    = [](){ const char* s=std::getenv("AMOR_PG_IREFC");    return s?atof(s):0.5; }();
    // F6 (AMOR_PG_SHAPE): dense potential-shaped reward weight (Ng-Harada-Russell). Default 0.0 = sparse best-drop (== current).
    static const double _pg_shape    = [](){ const char* s=std::getenv("AMOR_PG_SHAPE"); return s?atof(s):0.0; }();
    // F7 (AMOR_RWD_WHITEN): variance-normalized advantage in the PGPE update. Default 0 = EMA-baseline only (== current).
    static const int    _rwd_whiten  = [](){ const char* s=std::getenv("AMOR_RWD_WHITEN"); return s?atoi(s):0; }();
    // F8 (AMOR_PG_ANTITHETIC): paired +/-eps PGPE (common-mode rejection). Default 0 = single-sided (== current).
    static const int    _pg_antithetic = [](){ const char* s=std::getenv("AMOR_PG_ANTITHETIC"); return s?atoi(s):0; }();
    // F10 (AMOR_PG_SSCHED): annealed sigma_k + freeze + sigma-bias fix. Default 0 = constant sigma (== current).
    static const int    _pg_ssched = [](){ const char* s=std::getenv("AMOR_PG_SSCHED"); return s?atoi(s):0; }();
    static const double _pg_sig0   = [](){ const char* s=std::getenv("AMOR_PG_SIG0"); return s?atof(s):0.90; }();
    static const double _pg_sigt   = [](){ const char* s=std::getenv("AMOR_PG_SIGT"); return s?atof(s):0.08; }();
    static const double _pg_frz    = [](){ const char* s=std::getenv("AMOR_PG_FRZ");  return s?atof(s):0.34; }();
    // F2 (AMOR_ACCEPT_FLOOR): ramped anchor delta-floor (no-regret ratchet; shared modes 9 & 10). Default 0 = off.
    static const int    _accept_floor_on = [](){ const char* s=std::getenv("AMOR_ACCEPT_FLOOR"); return s?atoi(s):0; }();
    // GAP feature (theta-dim 5): LEARNED optimality-gap regime signal. g_feat = clamp((ref - LB)/(init_gap), 0, 1):
    // 1 at the start (far from the lower bound => open delta / explore), 0 near the LB (converged => delta->greedy,
    // protecting anytime AUC). One policy opens delta wherever the gap is high, with NO warm-start reach problem.
    // AMOR_GAP_W (default 0.0): warm-starts _spsa_th[5]. 0.0 => weight 0 => g_feat term contributes exactly 0 =>
    // the delta-head is byte-identical to the pre-gap binary (do-no-harm invariance). A positive value opens delta
    // with the gap from iteration 1 without waiting for the learner. AMOR_GAP_USEBEST (default 1): ref=_best_cost;
    // if 0, ref=sum_of_costs (current working cost). The gap dim is a FIXED warm-start weight in the default
    // spsa/pg-PGPE paths (NOT SPSA-perturbed: kept out of _spsa_dim; NOT PGPE-perturbed: _pg_eps[5]==0, so ZERO
    // extra rand() draws) -> this is what guarantees the theta5=0 run is byte-identical. The soft-acceptance
    // REINFORCE ablation (AMOR_PG_PGPE=0) does learn dim 5 via the added _pg_step[5] gradient.
    static const double _gap_w = [](){ const char* s=std::getenv("AMOR_GAP_W"); return s?atof(s):0.0; }();
    static const int    _gap_usebest = [](){ const char* s=std::getenv("AMOR_GAP_USEBEST"); return s?atoi(s):1; }();
    // AMOR_STAG>0: hybrid greedy-early / explore-late. Stay PURE GREEDY until no new global best for
    // STAG iterations (stagnation), then enable the non-greedy criterion. Removes early-time regression
    // (descends greedily like SOTA) while still escaping the late-time stagnation greedy gets stuck in.
    static const int _stag = [](){ const char* s=std::getenv("AMOR_STAG"); return s?std::max(0,atoi(s)):0; }();
    std::vector<int> _lahc_hist((size_t)std::max(1,_lahc_L), sum_of_costs);
    int _lahc_v = 0;
    int _best_cost = sum_of_costs;
    int _iters_since_improve = 0;
    int _run_best = sum_of_costs;                 // best working-cost ever visited (for AMOR_BEST_RETURN)
    std::vector<Path> _best_paths;                // snapshot of the best incumbent's paths (only if _best_return)
    // bandit arms: greedy / rr2 / rr5 / rr10 / ta5  (useBest=anchor to best vs current; delta=slack)
    const int _NARM = 5;
    int    _arm_useBest[5] = {0, 1, 1, 1, 0};
    double _arm_delta[5]   = {0, 2, 5, 10, 5};
    double _arm_val[5]     = {0, 0, 0, 0, 0};
    int    _arm_n[5]       = {0, 0, 0, 0, 0};
    int _cur_arm = 0, _win_count = 0, _win_start_best = sum_of_costs;
    // SPSA state: theta, current Rademacher perturbation, two-window phase, window bookkeeping.
    // Unified 5-param policy theta = [t0,t1,t2 | t3,t4]:
    //   tolerance  delta_eff = softplus(t0 + t1*s + t2*u)            (how much uphill to tolerate)
    //   anchor     lambda    = clip(t3 + t4*s, 0, 1)                 (0=best-ref/rr ... 1=current-ref/ta)
    //   accept iff cand <= [best + lambda*(current-best)] + delta_eff
    // This ONE policy subsumes greedy/rr/ta/adapt/adaptsw as special theta. Init ALL ZERO => delta_eff=
    // softplus(0)~0.69, lambda=0 => starts ~GREEDY (== SOTA LNS2, the honest baseline); SPSA must LEARN to
    // open exploration online. Worst case it stays ~greedy (no loss); per-run it tunes all 5.
    double _spsa_th[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // dim 5 = LEARNED optimality-gap weight (init 0 => invariance)
    int    _spsa_delta[6] = {1, 1, 1, 1, 1, 1};
    int    _spsa_phase = 0;                 // 0 = measuring theta+cD, 1 = measuring theta-cD
    double _spsa_Rplus = 0.0;
    int    _spsa_wc = 0, _spsa_wstart = sum_of_costs;
    // policy-gradient (mode 10) state: reuses _spsa_th[5] as theta.
    // FIX2 = actor-critic with eligibility traces: _pg_step = THIS-iter grad-log-pi, _pg_e = decaying trace,
    // _pg_gbatch = trace-credited gradient accumulated for a batched Adam step. Replaces the broken window-REINFORCE
    // (which credited uphill decisions with a windowed best-drop dominated by downhill improvements -> ~zero/negative
    // correlation -> delta never grew). Now an escape-to-new-best rewards the tolerant decisions that PRECEDED it.
    double _pg_m[6] = {0,0,0,0,0,0}, _pg_v[6] = {0,0,0,0,0,0}, _pg_glogp[6] = {0,0,0,0,0,0};
    double _pg_e[6] = {0,0,0,0,0,0}, _pg_gbatch[6] = {0,0,0,0,0,0}, _pg_step[6] = {0,0,0,0,0,0};
    double _pg_baseline = 0.0; int _pg_t = 0, _pg_wc = 0, _pg_wstart = sum_of_costs;
    // FIX1 warm-start: start at an rr-like criterion (delta=_pg_dinit via softplus^-1, lambda~0 best-anchored via t3=-3)
    // instead of greedy. Gives pg a safe lower bound (no cold-start regression) so the gradient only fine-tunes.
    // F4: with KNORM>=1 the warm-start delta is a FRACTION (_pg_dfrac) of the per-agent normalizer, not absolute _pg_dinit.
    if (_accept_init == 10 && _pg_dinit > 0.0) { double _d0 = (_pg_knorm >= 1) ? _pg_dfrac : _pg_dinit;
        _spsa_th[0] = std::log(std::exp(_d0) - 1.0); _spsa_th[3] = -3.0; _spsa_th[1] = _pg_t1init; }
    static const double _pg_lam = [](){ const char* s = std::getenv("AMOR_PG_LAM"); return s ? atof(s) : 0.0; }();  // L2 pull to warm-start (no-regret/do-no-harm anchor)
    if (_accept_init == 9 && _spsa_dinit > 0.0) _spsa_th[0] = std::log(std::exp(_spsa_dinit) - 1.0); // F1: warm-start SPSA theta0 AT rr anchor
    _spsa_th[5] = _gap_w;   // GAP: warm-start the learned optimality-gap weight (default 0.0 => byte-identical); applies to BOTH spsa (mode 9) & pg (mode 10)
    double _pg_w0[6]; for (int i = 0; i < 6; i++) _pg_w0[i] = _spsa_th[i];   // captured warm-start = the criterion pg must not drift away from
    // PGPE (FIX5) per-window perturbed params: theta_w = theta + eps, eps ~ N(0, sigma^2). Gaussian via Box-Muller.
    double _pg_eps[6] = {0,0,0,0,0,0}, _pg_thw[6] = {0,0,0,0,0,0};
    int    _pg_phase = 0; double _pg_Rplus = 0.0;                          // F8 antithetic pairing (+eps window then -eps window)
    double _pg_r2 = 0.0;                                                   // F7 whitening: second-moment EMA of the window reward
    double _pg_sig_w = _pg_sigma;                                          // F10: the sigma that GENERATED the current eps (gradient divides by THIS)
    double _pol_thbest[6]; for (int i = 0; i < 6; i++) _pol_thbest[i] = _spsa_th[i]; // F9/F10 freeze-commit: best-incumbent policy snapshot
    double _pol_last_improve_rt = runtime;                                 // F9/F10: runtime of the last new incumbent (stall clock)
    double _rwd_acc = 0.0; int _rwd_n = 0;                                 // F6 dense-reward window accumulator
    auto _pg_gauss = [](){ double u1=((double)rand()+1.0)/((double)RAND_MAX+2.0), u2=(double)rand()/(double)RAND_MAX;
                           return std::sqrt(-2.0*std::log(u1))*std::cos(6.283185307179586*u2); };
    if (_accept_init == 10 && _pg_pgpe) {
        _pg_sig_w = _pg_ssched ? _pg_sig0 : _pg_sigma;                     // F10: initial sigma = sigma0 (annealed at p~0) else constant sigma
        for (int i=0;i<5;i++) { _pg_eps[i] = _pg_sig_w * _pg_gauss(); _pg_thw[i] = _spsa_th[i] + _pg_eps[i]; }
        _pg_eps[5] = 0.0; _pg_thw[5] = _spsa_th[5];                        // GAP dim: fixed weight, NOT PGPE-perturbed (no _pg_gauss => no extra rand() => byte-identical stream)
    }
    while (runtime < time_limit && iteration_stats.size() <= num_of_iterations)
    {
        runtime =((fsec)(Time::now() - start_time)).count();
        if(screen >= 1)
            validateSolution();
        if (ALNS)
            chooseDestroyHeuristicbyALNS();

        switch (destroy_strategy)
        {
            case RANDOMWALK:
                succ = generateNeighborByRandomWalk();
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
                succ = true;
                break;
            default:
                cerr << "Wrong neighbor generation strategy" << endl;
                exit(-1);
        }
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

        // ── COUNTERFACTUAL ORDER-PROBE logger (AMOR_ORDER_PROBE=<csv>) ──────────────────────────
        // Per replan iteration: try each repair-ORDER arm {random, longest-haul, most-delayed} on the SAME
        // destroyed neighborhood, record each arm's realized reduction, REVERT to the post-destroy state,
        // then let the real repair proceed unchanged. Answers Smoke-0/1: does PER-NEIGHBORHOOD context
        // predict the best order beyond a per-map constant (=> genuine contextual bandit vs ornamental gate)?
        // Non-invasive: uses a LOCAL RNG for the random arm so the global rand() stream (real trajectory) is
        // untouched. ~3x PP per logged iter — a LOGGING smoke, not deployment.
        { const char* _op = std::getenv("AMOR_ORDER_PROBE");
          if (_op && replan_algo_name == "PP" && !iteration_stats.empty()
              && !neighbor.old_paths.empty() && neighbor.agents.size() >= 2)
          {
            if (!g_ord_probe_open) { g_ord_probe_open = true; g_ord_probe_fp = fopen(_op, "a");
                if (g_ord_probe_fp && ftell(g_ord_probe_fp) == 0)
                    fprintf(g_ord_probe_fp, "iter,nsize,total_delay,max_share,cong_c,iter_frac,red_rand,red_long,red_mdel\n"); }
            if (g_ord_probe_fp) {
                int nsz = (int)neighbor.agents.size();
                long tot_delay = 0, mx = 0;
                for (int a : neighbor.agents) { long d = agents[a].getNumOfDelays(); tot_delay += d; if (d > mx) mx = d; }
                double max_share = tot_delay > 0 ? (double)mx / tot_delay : 0.0;
                double cong_c = (!agents.empty()) ? (double)(initial_sum_of_costs - sum_of_distances) / agents.size() : 0.0;
                double iter_frac = (time_limit > 0) ? runtime / time_limit : 0.0;
                double base = (double)neighbor.old_sum_of_costs;
                auto hd = [&](int a){ return agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]; };
                auto probe = [&](int ord)->double {                       // plan neighbor in `ord`, return reduction frac, revert
                    std::vector<int> sa = neighbor.agents;
                    if (ord == 1)      std::sort(sa.begin(), sa.end(), [&](int a,int b){ return hd(a) > hd(b); });                          // longest-haul first
                    else if (ord == 3) std::sort(sa.begin(), sa.end(), [&](int a,int b){ return agents[a].getNumOfDelays() > agents[b].getNumOfDelays(); }); // most-delayed first
                    else { std::mt19937 _g(1315423911u * (unsigned)(iteration_stats.size() + 1)); std::shuffle(sa.begin(), sa.end(), _g); } // random (LOCAL rng, non-invasive)
                    ConstraintTable ct(instance.num_of_cols, instance.map_size, &path_table);
                    int sum = 0; std::vector<int> planned; bool ok = true;
                    for (int id : sa) { Path pth = agents[id].path_planner->findPath(ct);
                        if (pth.empty()) { ok = false; break; }
                        agents[id].path = pth; sum += (int)pth.size() - 1;
                        path_table.insertPath(agents[id].id, pth); planned.push_back(id); }
                    for (int id : planned) path_table.deletePath(agents[id].id, agents[id].path);
                    for (size_t i = 0; i < neighbor.agents.size(); i++) agents[neighbor.agents[i]].path = neighbor.old_paths[i];
                    return ok ? (base - (double)sum) / std::max(1.0, base) : -1.0;
                };
                double rr = probe(0), rl = probe(1), rm = probe(3);
                fprintf(g_ord_probe_fp, "%d,%d,%ld,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f\n",
                        (int)iteration_stats.size(), nsz, tot_delay, max_share, cong_c, iter_frac, rr, rl, rm);
                fflush(g_ord_probe_fp);
            }
          }
        }

        // ── CACo-LNS selective exact-repair gate (modification layered on top of the OFFICIAL runEECBS) ──────────
        // Zero-invasive: with all knobs unset this whole block is skipped and the dispatch below is
        // byte-for-byte the official one. When AMOR_NORM_THRESH>=0 (or LNS_FORCE_EECBS is set) AND the
        // CLI replanAlgo is PP, we run the OFFICIAL runEECBS() on the destroyed neighborhood instead of
        // PP, but ONLY when the neighborhood is a scale-invariant congestion bottleneck (per-agent avg
        // delay / map avg path length > threshold) and is small enough to be EECBS-feasible. K-shrink
        // keeps only the K most-delayed agents for exact repair (the rest are restored as fixed
        // obstacles) so the ECBS sub-solve never explodes at high density. We do NOT touch runEECBS().
        //   AMOR_NORM_THRESH  : >=0 => fire EECBS when norm_delay > this dimensionless ratio (scale-invariant)
        //   LNS_FORCE_EECBS   : present => fire EECBS every (feasible) iteration (== always-exact arm)
        //   AMOR_EECBS_SIZE_MAX: neighborhood-size feasibility cap for firing EECBS (default 16)
        //   AMOR_EECBS_MAX_K  : >=2 => shrink the exact-repair set to the K most-delayed agents
        //   AMOR_SHRINK_THRESH: <0 => always shrink (when K set); >=0 => shrink only when avg_delay > this
        //   AMOR_EECBS_TL     : per-call EECBS wall-clock cap in seconds (default 2.0)
        auto _cgetd = [](const char* k, double def){ const char* v = std::getenv(k); return v ? atof(v) : def; };
        static const double G_NORM_THRESH   = _cgetd("AMOR_NORM_THRESH", -1.0);
        static const double G_EECBS_SIZEMAX = _cgetd("AMOR_EECBS_SIZE_MAX", 16.0);
        static const double G_EECBS_MAX_K   = _cgetd("AMOR_EECBS_MAX_K", 0.0);
        static const double G_SHRINK_THRESH = _cgetd("AMOR_SHRINK_THRESH", -1.0);
        static const double G_EECBS_TL      = _cgetd("AMOR_EECBS_TL", 2.0);
        static const bool   G_FORCE_EECBS   = (std::getenv("LNS_FORCE_EECBS") != nullptr);
        bool _sel_eecbs = false;
        if (replan_algo_name == "PP" && (G_FORCE_EECBS || G_NORM_THRESH >= 0.0))
        {
            int _nsz = (int)neighbor.agents.size();
            int _ndelay = 0; for (int _a : neighbor.agents) _ndelay += agents[_a].getNumOfDelays();
            double _avgd = _nsz > 0 ? (double)_ndelay / _nsz : 0.0;
            double _map_avg_path = agents.empty() ? 0.0 : (double)initial_sum_of_costs / agents.size();
            double _norm = (_map_avg_path > 1e-9) ? (_avgd / _map_avg_path) : 0.0;
            double _now_rt = ((fsec)(Time::now() - start_time)).count();
            bool _gate = (_nsz >= 2 && _nsz <= (int)G_EECBS_SIZEMAX && (time_limit - _now_rt) > 0.6);
            bool _trigger = G_FORCE_EECBS || (_norm > G_NORM_THRESH);
            _sel_eecbs = _trigger && _gate;
        }
        if (_sel_eecbs)
        {
            // Density-adaptive K-shrink: exact-repair only the K most-delayed agents; restore the rest to
            // their (already-saved, PP-path) original paths as fixed obstacles for ECBS to plan around.
            int _K = (int)G_EECBS_MAX_K;
            int _nsz0 = (int)neighbor.agents.size();
            int _nd0 = 0; for (int _a : neighbor.agents) _nd0 += agents[_a].getNumOfDelays();
            double _avgd0 = _nsz0 > 0 ? (double)_nd0 / _nsz0 : 0.0;
            bool _hard = (G_SHRINK_THRESH < 0.0 || _avgd0 > G_SHRINK_THRESH);
            if (_K >= 2 && _hard && _nsz0 > _K
                && (int)neighbor.old_paths.size() == _nsz0 && !neighbor.old_paths[0].empty())
            {
                vector<pair<int,int>> _rank; _rank.reserve(_nsz0);
                for (int _j = 0; _j < _nsz0; _j++)
                    _rank.emplace_back(agents[neighbor.agents[_j]].getNumOfDelays(), _j);
                sort(_rank.begin(), _rank.end(), std::greater<pair<int,int>>());
                vector<char> _keep(_nsz0, 0);
                for (int _j = 0; _j < _K; _j++) _keep[_rank[_j].second] = 1;
                vector<int> _na; vector<Path> _nop;
                for (int _j = 0; _j < _nsz0; _j++)
                {
                    int _aid = neighbor.agents[_j];
                    if (_keep[_j]) { _na.push_back(_aid); _nop.push_back(neighbor.old_paths[_j]); }
                    else { agents[_aid].path = neighbor.old_paths[_j]; path_table.insertPath(_aid, agents[_aid].path); }
                }
                neighbor.agents = _na; neighbor.old_paths = _nop;
                neighbor.old_sum_of_costs = 0;
                for (auto& _p : neighbor.old_paths) neighbor.old_sum_of_costs += (int)_p.size() - 1;
            }
            double _saved_rtl = replan_time_limit;
            double _now_rt2 = ((fsec)(Time::now() - start_time)).count();
            replan_time_limit = max(0.5, min(time_limit - _now_rt2 - 0.05, G_EECBS_TL));
            succ = runEECBS();          // OFFICIAL EECBS, reused verbatim
            replan_time_limit = _saved_rtl;
        }
        else if (replan_algo_name == "EECBS")
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
            if (neighbor.old_sum_of_costs > neighbor.sum_of_costs )
                destroy_weights[selected_neighbor] =
                        reaction_factor * (neighbor.old_sum_of_costs - neighbor.sum_of_costs) / neighbor.agents.size()
                        + (1 - reaction_factor) * destroy_weights[selected_neighbor];
            else
                destroy_weights[selected_neighbor] =
                        (1 - decay_factor) * destroy_weights[selected_neighbor];
        }
        // ── Learned/bandit destroy seed-selection reward update (reward = realized neighborhood improvement) ──
        if (succ)
        {
            if (g_sd_last >= 0 && !neighbor.agents.empty())   // mode 10: contextual-bandit REINFORCE
            {
                double r_neigh = (double)(neighbor.old_sum_of_costs - neighbor.sum_of_costs) / (double)neighbor.agents.size();
                double r_seed = r_neigh;   // OPT-1 de-confound: credit the SEED's OWN delay reduction (old-len - new-len)
                if (neighbor.old_paths.size() == neighbor.agents.size())
                    for (int i = 0; i < (int)neighbor.agents.size(); i++)
                        if (neighbor.agents[i] == g_sd_last) {
                            r_seed = (double)((int)neighbor.old_paths[i].size() - (int)agents[g_sd_last].path.size()); break; }
                double r = g_sd_beta * r_seed + (1.0 - g_sd_beta) * r_neigh;   // blend seed-own (attributable) + neighborhood (global)
                double var = g_sd_r2 - g_sd_baseline*g_sd_baseline; if (var < 1e-9) var = 1e-9;
                double A = (r - g_sd_baseline) / std::sqrt(var); g_sd_t++;   // WHITENED advantage (scale-free -> stable gradient, less drift)
                for (int k = 0; k < 4; k++) {
                    double grad = A * (g_sd_phi_chosen[k] - g_sd_phibar[k]) / g_sd_tau;
                    g_sd_m[k] = 0.9*g_sd_m[k] + 0.1*grad; g_sd_v[k] = 0.999*g_sd_v[k] + 0.001*grad*grad;
                    double mh = g_sd_m[k]/(1.0-std::pow(0.9,g_sd_t)), vh = g_sd_v[k]/(1.0-std::pow(0.999,g_sd_t));
                    g_sd_theta[k] += g_sd_alpha*mh/(std::sqrt(vh)+1e-8);
                    g_sd_theta[k] -= g_sd_alpha*g_sd_lam*(g_sd_theta[k]-g_sd_W0[k]);   // L2 pull to warm-start (do-no-harm)
                    if (g_sd_theta[k] < -10.0) g_sd_theta[k] = -10.0; if (g_sd_theta[k] > 10.0) g_sd_theta[k] = 10.0;
                }
                g_sd_baseline = 0.9*g_sd_baseline + 0.1*r; g_sd_r2 = 0.9*g_sd_r2 + 0.1*r*r;
                g_sd_yema[g_sd_last] = 0.9*g_sd_yema[g_sd_last] + 0.1*r_seed;   // per-agent value = its OWN realized delay reduction
                g_sd_touched[g_sd_last] = (int)g_sd_iter; g_sd_iter++; g_sd_last = -1;
            }
            // B1 FIX: Beta-bandit (mode 20/30) reward update MOVED OUT of this if(succ) block — see below.
            // Gating on if(succ) dropped every FAILURE (greedy runPP returns succ=false on non-improve),
            // so beta never grew and the bandit could not down-weight bad seeds (anti-learning).
            if (g_ro_have)        // repair-order Plackett-Luce REINFORCE (reward = -relative slack; lower slack = better order)
            {
                double r = -(double)(neighbor.sum_of_costs - g_ro_lb) / std::max(1.0, g_ro_lb);
                double rvar = g_ro_r2 - g_ro_baseline*g_ro_baseline; if (rvar < 1e-12) rvar = 1e-12;
                double A = (r - g_ro_baseline) / std::sqrt(rvar); g_ro_t++;   // whitened advantage
                for (int f=0;f<2;f++){ double g = A * g_ro_grad[f];
                    g_ro_m[f]=0.9*g_ro_m[f]+0.1*g; g_ro_v[f]=0.999*g_ro_v[f]+0.001*g*g;
                    double mh=g_ro_m[f]/(1.0-std::pow(0.9,g_ro_t)), vh=g_ro_v[f]/(1.0-std::pow(0.999,g_ro_t));
                    g_ro_theta[f]+=g_ro_alpha*mh/(std::sqrt(vh)+1e-8);
                    if(g_ro_theta[f]<-10.0)g_ro_theta[f]=-10.0; if(g_ro_theta[f]>10.0)g_ro_theta[f]=10.0; }
                g_ro_baseline = 0.9*g_ro_baseline + 0.1*r; g_ro_r2 = 0.9*g_ro_r2 + 0.1*r*r; g_ro_have = false;
            }
            // B1 FIX: repair-bandit update MOVED OUT of this if(succ) block (see below) — gating on succ
            // dropped every FAILED repair, so arms that fail often were never penalized (biased values).
            if (g_pr_last >= 0)   // PROBE: log (features, rewards) for the offline residual-signal test
            {
                if (!g_pr_open) { g_pr_open = true; const char* p = std::getenv("AMOR_LOG");
                    if (p) { g_pr_fp = fopen(p, "a"); if (g_pr_fp && ftell(g_pr_fp)==0)
                        fprintf(g_pr_fp, "logdelay,reldelay,stretch,staleness,invdeg,corridor,r_neigh,r_seed,walk_ok,nsize\n"); } }
                if (g_pr_fp) {
                    double r_neigh = (double)(neighbor.old_sum_of_costs - neighbor.sum_of_costs) / (double)neighbor.agents.size();
                    double r_seed = r_neigh;
                    if (neighbor.old_paths.size() == neighbor.agents.size())
                        for (int i=0;i<(int)neighbor.agents.size();i++) if (neighbor.agents[i]==g_pr_last) {
                            r_seed = (double)((int)neighbor.old_paths[i].size() - (int)agents[g_pr_last].path.size()); break; }
                    fprintf(g_pr_fp, "%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,1,%d\n",
                            g_pr_phi[0],g_pr_phi[1],g_pr_phi[2],g_pr_phi[3],g_pr_phi[4],g_pr_phi[5],r_neigh,r_seed,(int)neighbor.agents.size());
                    fflush(g_pr_fp);
                }
                g_pr_last = -1;
            }
            if (g_lt_last >= 0)   // LinTS/LinUCB posterior update (reward-standardized, Sherman-Morrison, O(d^2))
            {
                double r = (double)(neighbor.old_sum_of_costs - neighbor.sum_of_costs) / (double)neighbor.agents.size();
                g_lt_rmean = 0.9*g_lt_rmean + 0.1*r; g_lt_r2 = 0.9*g_lt_r2 + 0.1*r*r;
                double rt = (r - g_lt_rmean) / std::sqrt(std::max(1e-9, g_lt_r2 - g_lt_rmean*g_lt_rmean));
                double w[6]={0}; for(int i=0;i<6;i++) for(int j=0;j<6;j++) w[i]+=g_lt_Ainv[i][j]*g_lt_xlast[j];
                double denom=1.0; for(int j=0;j<6;j++) denom+=g_lt_xlast[j]*w[j];
                for(int i=0;i<6;i++) for(int j=0;j<6;j++) g_lt_Ainv[i][j]-=w[i]*w[j]/denom;
                for(int j=0;j<6;j++) g_lt_b[j]+=rt*g_lt_xlast[j];
                for(int i=0;i<6;i++){ double t=0; for(int j=0;j<6;j++) t+=g_lt_Ainv[i][j]*g_lt_b[j]; g_lt_theta[i]=t; }
                g_lt_last=-1;
            }
        }
        // B1 FIX: Beta-bandit reward recorded EVERY iteration a seed was chosen (win AND loss), using the
        // REPAIR outcome (faithful to ADDRESS Alg.2 l.9-14). Runs BEFORE the acceptance revert so the reward
        // reflects repair quality, not the accept decision. Zero cost (a couple comparisons).
        {
            bool _bandit_improved = succ && (neighbor.old_sum_of_costs - neighbor.sum_of_costs > 0);
            if (g_ts_last >= 0) {   // mode 20 ADDRESS Beta
                if (_bandit_improved) g_ts_a[g_ts_last] += 1.0; else g_ts_b[g_ts_last] += 1.0;
                g_ts_last = -1;
            }
            if (g_tk_col >= 0) {    // mode 30 TACKLE counterfactual table
                if (_bandit_improved) g_tk_alpha[g_tk_intent][g_tk_col] += 1; else g_tk_beta[g_tk_intent][g_tk_col] += 1;
                g_tk_col = -1;
            }
            if (g_cu_last >= 0) {   // AMOR-CUCB reward: observed post-repair pre-acceptance, EVERY pull.
                double _r;
                if (g_cu_reward == 1) {   // slack-fraction (fallback reward per pre-registration)
                    double _sl = std::max(1.0, (double)neighbor.old_sum_of_costs - g_cu_lb);
                    _r = ((double)neighbor.old_sum_of_costs - (double)neighbor.sum_of_costs) / _sl;
                    if (_r < 0.0) _r = 0.0; if (_r > 1.0) _r = 1.0;
                } else if (g_cu_reward == 2) {   // V5 rawdelta: Bernoulli(clip((old-new)/16)) — magnitude-aware, mean = scaled raw improvement
                    double p = ((double)neighbor.old_sum_of_costs - (double)neighbor.sum_of_costs) / 16.0;
                    if (p < 0) p = 0; if (p > 1) p = 1;
                    _r = (amor_u01((unsigned)(neighbor.old_sum_of_costs*31u + (unsigned)g_cu_t)) < p) ? 1.0 : 0.0;
                } else                    // completion indicator (primary): repair completed AND improving
                    _r = (succ && neighbor.old_sum_of_costs > neighbor.sum_of_costs) ? 1.0 : 0.0;
                if (g_cu_mode == 4) {   // roulette segment scores (33/13/0, see decl comment)
                    g_cu_sc[g_cu_last] += (succ && neighbor.old_sum_of_costs > neighbor.sum_of_costs) ? 33.0 : (succ ? 13.0 : 0.0);
                    g_cu_scn[g_cu_last]++; }
                {   // Phase-0 mechanism log (env-gated, zero overhead when off): per-pull arm/reward/raw-delta/incumbent
                    static const bool _culog = std::getenv("AMOR_CU_LOG") != nullptr;
                    if (_culog) std::cerr << "CULOG " << g_cu_t << " " << g_cu_last << " " << _r << " "
                                          << neighbor.old_sum_of_costs << " " << neighbor.sum_of_costs << " "
                                          << (succ?1:0) << " " << sum_of_costs << "\n"; }
                g_cu_sum[g_cu_last] += _r; g_cu_sq[g_cu_last] += _r*_r; g_cu_n[g_cu_last]++; g_cu_S += _r; g_cu_last = -1;
            }
            if (g_rb_last >= 0) {   // repair-order bandit (B1 fix): update on EVERY pull, win AND loss.
                // reward = fraction of the neighborhood's recoverable slack actually recovered (0 on failed
                // repair since sum was reverted to old) — difficulty-normalized, scale-free across maps.
                double _slack = std::max(1.0, (double)neighbor.old_sum_of_costs - g_rb_lb);
                double _r = ((double)neighbor.old_sum_of_costs - (double)neighbor.sum_of_costs) / _slack;
                if (_r < -1.0) _r = -1.0; if (_r > 1.0) _r = 1.0;
                g_rb_val[g_rb_last] += g_rb_alpha * (_r - g_rb_val[g_rb_last]); g_rb_n[g_rb_last]++;
                g_rb_last = -1;
            }
        }
        // ── Zero-overhead non-greedy acceptance decision (AMOR_ACCEPT) ──────────────
        // PP produced a COMPLETE neighborhood (possibly costlier). Decide whether to keep it by a
        // zero-overhead criterion; on reject, revert to old paths (global solution unchanged). No search.
        if (_accept_init != 0 && succ && !neighbor.old_paths.empty())
        {
            int cand = sum_of_costs + (neighbor.sum_of_costs - neighbor.old_sum_of_costs); // global cost if kept
            // F9: SPSA step (a_k) and perturbation (c_k) schedule + freeze-best commit. Default (SCHED=0) = constant (== current).
            double _sp_ck = _spsa_c, _sp_ak = _spsa_a;
            if (_accept_init == 9 && _spsa_sched) {
                double _p = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
                _sp_ak = _spsa_a0 * std::pow(_spsa_at / _spsa_a0, _p);      // anneal step a0 -> aT (front-load far travel from greedy init)
                _sp_ck = _spsa_c  * std::pow(_spsa_ct / _spsa_c,  _p);      // anneal perturbation c -> cT
                double _stall = (time_limit > 0) ? (runtime - _pol_last_improve_rt) / time_limit : 0.0;
                if (_stall > _spsa_frz) { _sp_ak = 0.0; _sp_ck = 0.0; for (int i = 0; i < 6; i++) _spsa_th[i] = _pol_thbest[i]; } // freeze + commit to best policy
            }
            bool allow_explore = (_stag <= 0) || (_iters_since_improve >= _stag); // hybrid gate (pure greedy until stuck)
            bool accept;
            bool _switched_greedy = (_switch_frac >= 0.0 && time_limit > 0 && runtime > _switch_frac * time_limit);
            if (cand <= sum_of_costs) accept = true;                            // always keep improvements (greedy)
            else if (_switched_greedy) accept = false;                          // oracle time-switch: greedy after f*T
            else if (!allow_explore) accept = false;                            // greedy phase: reject any uphill
            else if (_accept_init == 1) accept = (cand <= _lahc_hist[_lahc_v]);  // LAHC: no worse than L steps ago
            else if (_accept_init == 2) accept = (cand <= _best_cost + (int)_rr_delta); // record-to-record travel (vs best)
            else if (_accept_init == 4) accept = (cand <= sum_of_costs + (int)_rr_delta); // threshold accepting (vs current)
            else if (_accept_init == 11) { double tau = (time_limit > 0) ? runtime / time_limit : 0.0; if (tau > 1) tau = 1;
                double L = (double)_best_cost + _gd_d0 * (1.0 - tau); accept = (cand <= (int)L); } // great-deluge: budget-tied decreasing slack
            else if (_accept_init == 5) { int de = (int)std::min(_adapt_cap, _adapt_floor + _adapt_slope * _iters_since_improve); accept = (cand <= _best_cost + de); }    // adaptive-delta vs best (Cand 1)
            else if (_accept_init == 6) { int de = (int)std::min(_adapt_cap, _adapt_floor + _adapt_slope * _iters_since_improve); accept = (cand <= sum_of_costs + de); } // adaptive-delta vs current (pure ta-adaptive arm)
            else if (_accept_init == 7) { int de = (int)std::min(_adapt_cap, _adapt_floor + _adapt_slope * _iters_since_improve); // Cand 2: shallow->current (ta relocate), deep->best (rr refine)
                int ref = (_iters_since_improve < _adapt_switch) ? sum_of_costs : _best_cost; accept = (cand <= ref + de); }
            else if (_accept_init == 8) { int ref = _arm_useBest[_cur_arm] ? _best_cost : sum_of_costs; accept = (cand <= ref + (int)_arm_delta[_cur_arm]); } // bandit-selected arm
            else if (_accept_init == 9) {                                                   // SPSA unified continuous policy
                double s_feat = std::min(1.0, (double)_iters_since_improve / _spsa_snorm);
                double u_feat = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
                int    _gref  = _gap_usebest ? _best_cost : sum_of_costs;                   // GAP feature (theta-dim 5): live optimality gap to the LB
                double g0abs  = std::max(1, initial_sum_of_costs - sum_of_distances);       // initial gap (const per run)
                double g_feat = std::min(1.0, std::max(0.0, (double)(_gref - sum_of_distances) / g0abs)); // 1 at start -> 0 near LB
                double sign = (_spsa_phase == 0) ? 1.0 : -1.0;                              // evaluate theta +/- c*Delta
                double th[6]; for (int i = 0; i < 6; i++) th[i] = (i < _spsa_dim) ? _spsa_th[i] + sign * _sp_ck * _spsa_delta[i] : ((i == 5) ? _spsa_th[5] : 0.0); // dim 5 = fixed gap weight (never SPSA-perturbed since _spsa_dim<=5)
                double z = th[0] + th[1] * s_feat + th[2] * u_feat + th[5] * g_feat;        // tolerance exponent (+ learned gap term)
                double de = (z > 30.0) ? z : std::log(1.0 + std::exp(z));                   // softplus
                if (de < 0) de = 0; if (de > _spsa_dmax) de = _spsa_dmax;
                if (_accept_floor_on) {   // F2: ramped anchor delta-floor (no-regret ratchet); SPSA lacks r_feat so compute it locally
                    double _rf = (_pg_iref > 0.0) ? std::min(1.0, (double)iteration_stats.size() / _pg_iref) : 1.0;
                    double _fl = _spsa_dinit * _rf; if (de < _fl) de = _fl;
                }
                double lam = th[3] + th[4] * s_feat;                                        // anchor blend best<->current
                if (lam < 0) lam = 0; if (lam > 1) lam = 1;
                int ref = _best_cost + (int)(lam * (double)(sum_of_costs - _best_cost));    // blended reference
                accept = (cand <= ref + (int)de);
            }
            else if (_accept_init == 10) {                                                  // policy-gradient acceptance (uphill only; cand<=cur already accepted above)
                double s_feat = (double)_iters_since_improve / ((double)_iters_since_improve + _spsa_snorm); // FIX3: soft-saturating stagnation (never pinned flat at 1 on hard maps)
                double u_feat = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
                int    _gref  = _gap_usebest ? _best_cost : sum_of_costs;                    // GAP feature (theta-dim 5): live optimality gap to the LB
                double g0abs  = std::max(1, initial_sum_of_costs - sum_of_distances);        // initial gap (const per run)
                double g_feat = std::min(1.0, std::max(0.0, (double)(_gref - sum_of_distances) / g0abs)); // 1 at start -> 0 near LB
                double kappa; // F4: 0=absolute(current), 1=best/N(legacy FIX4), 2=per-agent optimality gap
                if (agents.empty() || _pg_knorm == 0) kappa = 1.0;
                else if (_pg_knorm == 1) kappa = (double)_best_cost / (double)agents.size();
                else kappa = std::max(1.0, ((double)_best_cost - (double)sum_of_distances) / (double)agents.size());
                const double* TH = _pg_pgpe ? _pg_thw : _spsa_th;                            // PGPE acts with per-window perturbed params; soft acts with the mean
                double r_feat; // F5: 1=legacy fixed iref (current/DEFAULT), 0=flat r=1, 2=online projected iref
                if (_pg_irefmode == 1) r_feat = (_pg_iref > 0.0) ? std::min(1.0, (double)iteration_stats.size() / _pg_iref) : 1.0;
                else if (_pg_irefmode == 0) r_feat = 1.0;
                else { double _u = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
                       double _prj = (double)iteration_stats.size() / std::max(_u, 1e-3);
                       r_feat = std::min(1.0, (double)iteration_stats.size() / std::max(1.0, _pg_irefc * _prj)); }
                double zd = TH[0] * r_feat + TH[1] * s_feat + TH[2] * u_feat + TH[5] * g_feat; // delta opens only once iteration-rich (greedy early / on congested maps) (+ learned gap term)
                double sigd = 1.0 / (1.0 + std::exp(-zd));                                   // d softplus / dz
                double de = (zd > 30.0) ? zd : std::log(1.0 + std::exp(zd)); if (de > _spsa_dmax) de = _spsa_dmax;
                de *= kappa;                                                                 // tolerance in (optionally scaled) SOC units
                if (_accept_floor_on) {   // F2: ramped anchor delta-floor, POST-kappa. r_feat exists above.
                    // NOTE (untested stacking): with F4 KNORM!=0, de is in post-kappa units while _pg_dinit is absolute SOC;
                    // to stack, express the anchor as _pg_dinit/kappa. We ablate F2 and F4 as SEPARATE arms.
                    double _fl = _pg_dinit * r_feat; if (de < _fl) de = _fl;
                }
                double zl = TH[3] + TH[4] * s_feat;
                double lam = 1.0 / (1.0 + std::exp(-zl));
                double lamd = lam * (1.0 - lam);                                             // d sigmoid / dz
                double sb = (double)(sum_of_costs - _best_cost);
                double Tthr = (double)_best_cost + lam * sb + de;                            // threshold
                if (_pg_pgpe) {
                    accept = (cand <= (int)Tthr);                                            // FIX5: DETERMINISTIC within window (no per-iter stochastic penalty)
                } else {
                    double p = 1.0 / (1.0 + std::exp(-(Tthr - (double)cand) / _pg_tau));     // soft P(accept) (ablation)
                    int a = ((double)rand() / RAND_MAX < p) ? 1 : 0;
                    accept = (a == 1);
                    double coef = (a - p) / _pg_tau;                                         // d log pi / dT
                    _pg_step[0] = coef * sigd * kappa * r_feat;                              // THIS-iter grad-log-pi (eligibility trace assigns credit later)
                    _pg_step[1] = coef * sigd * kappa * s_feat;
                    _pg_step[2] = coef * sigd * kappa * u_feat;
                    _pg_step[3] = coef * sb * lamd;
                    _pg_step[4] = coef * sb * lamd * s_feat;
                    _pg_step[5] = coef * sigd * kappa * g_feat;                              // GAP dim grad-log-pi (soft-acceptance REINFORCE ablation learns dim 5)
                }
            }
            else accept = false;
            if (!accept) // revert neighborhood to old paths -> global solution unchanged
            {
                for (int i = 0; i < (int)neighbor.agents.size(); i++)
                    path_table.deletePath(neighbor.agents[i], agents[neighbor.agents[i]].path);
                for (int i = 0; i < (int)neighbor.agents.size(); i++)
                {
                    int a = neighbor.agents[i];
                    agents[a].path = neighbor.old_paths[i];
                    path_table.insertPath(agents[a].id, agents[a].path);
                }
                neighbor.sum_of_costs = neighbor.old_sum_of_costs; // makes the update below a no-op
            }
            int cur_after = sum_of_costs + (neighbor.sum_of_costs - neighbor.old_sum_of_costs);
            if (_accept_init == 1) { _lahc_hist[_lahc_v] = cur_after; _lahc_v = (_lahc_v + 1) % (int)_lahc_hist.size(); }
            int _pg_best_before = _best_cost;                                                 // FIX2: best before this iter's update (trace reward)
            if (cur_after < _best_cost) { _best_cost = cur_after; _iters_since_improve = 0;    // new global best -> reset stagnation
                if ((_accept_init == 9 && _spsa_sched) || (_accept_init == 10 && _pg_ssched)) { // F9/F10: snapshot best policy + reset stall clock
                    for (int i = 0; i < 6; i++) _pol_thbest[i] = _spsa_th[i]; _pol_last_improve_rt = runtime; }
            }
            else _iters_since_improve++;                                                      // stagnating
            if (_pg_shape != 0.0) {   // F6: dense potential-shaped reward (Ng-Harada-Russell, gamma=1 telescoping; optimum-preserving)
                double _rwd_scale = std::max(1, _pg_best_before);
                double _rwd_bdrop = (cur_after < _pg_best_before) ? (double)(_pg_best_before - cur_after) / _rwd_scale : 0.0;
                double _rwd_shape = ((double)(sum_of_costs - _pg_best_before) - (double)(cur_after - _best_cost)) / _rwd_scale; // Phi(s')-Phi(s)
                double _rwd_dense = _rwd_bdrop + _pg_shape * _rwd_shape;                       // SHAPE=0 -> this block skipped -> pure sparse
                _rwd_acc += _rwd_dense; _rwd_n++;
            }
            if (_accept_init == 8) {                                                          // bandit window bookkeeping
                _win_count++;
                if (_win_count >= _bandit_W) {
                    double reward = (double)(_win_start_best - _best_cost); if (reward < 0) reward = 0; // best-SOC drop this window
                    _arm_val[_cur_arm] += _bandit_alpha * (reward - _arm_val[_cur_arm]); _arm_n[_cur_arm]++;
                    int next = -1;
                    for (int a = 0; a < _NARM; a++) if (_arm_n[a] == 0) { next = a; break; }     // warmup: try each arm once
                    if (next < 0) {
                        if ((double)rand() / RAND_MAX < _bandit_eps) next = rand() % _NARM;        // explore
                        else { next = 0; for (int a = 1; a < _NARM; a++) if (_arm_val[a] > _arm_val[next]) next = a; } // exploit
                    }
                    _cur_arm = next; _win_count = 0; _win_start_best = _best_cost;
                }
            }
            if (_accept_init == 9) {                                                          // SPSA window: two-window gradient step
                _spsa_wc++;
                if (_spsa_wc >= _spsa_W) {
                    double R;                                                                 // F6: dense windowed-mean reward (keep sign) vs sparse best-drop (current)
                    if (_pg_shape != 0.0) R = (_rwd_n > 0) ? _rwd_acc / _rwd_n : 0.0;
                    else { R = (double)(_spsa_wstart - _best_cost) / (double)std::max(1, _spsa_wstart); if (R < 0) R = 0; }
                    if (_spsa_phase == 0) { _spsa_Rplus = R; _spsa_phase = 1; }               // measured theta+cD; now measure theta-cD
                    else {
                        double gsign = (_spsa_Rplus > R) ? 1.0 : ((_spsa_Rplus < R) ? -1.0 : 0.0); // ascend windowed reward
                        for (int i = 0; i < _spsa_dim; i++) {
                            _spsa_th[i] += _sp_ak * gsign * _spsa_delta[i];                    // F9: annealed/frozen step (== _spsa_a when SCHED off)
                            if (_spsa_th[i] < -10.0) _spsa_th[i] = -10.0;
                            if (_spsa_th[i] >  10.0) _spsa_th[i] =  10.0;
                        }
                        for (int i = 0; i < _spsa_dim; i++) _spsa_delta[i] = (rand() % 2) ? 1 : -1;   // fresh perturbation
                        _spsa_phase = 0;
                    }
                    _spsa_wc = 0; _spsa_wstart = _best_cost; _rwd_acc = 0; _rwd_n = 0;
                }
            }
            if (_accept_init == 10 && _pg_pgpe) {                                             // FIX5/F7/F8/F10: unified PGPE update (one window = one perturbed episode)
                _pg_wc++;
                if (_pg_wc >= _pg_W) {
                    double R;                                                                 // F6: dense windowed-mean reward (keep sign) vs sparse best-drop (current)
                    if (_pg_shape != 0.0) R = (_rwd_n > 0) ? _rwd_acc / _rwd_n : 0.0;
                    else { R = (double)(_pg_wstart - _best_cost) / (double)std::max(1, _pg_wstart); if (R < 0) R = 0; }
                    double _p = (time_limit > 0) ? std::min(1.0, runtime / time_limit) : 0.0;
                    bool _frozen = false;                                                     // F10: freeze + commit to best policy after a stall
                    if (_pg_ssched) { double _stall = (time_limit > 0) ? (runtime - _pol_last_improve_rt) / time_limit : 0.0;
                                      if (_stall > _pg_frz) _frozen = true; }
                    if (_pg_antithetic && _pg_phase == 0) {                                   // F8: +eps window done; now measure -eps with the SAME eps
                        _pg_Rplus = R; _pg_phase = 1;
                        for (int i = 0; i < 6; i++) _pg_thw[i] = _spsa_th[i] - _pg_eps[i];    // act with theta - eps (gap dim: eps[5]==0 => thw[5]=th[5])
                        _pg_wc = 0; _pg_wstart = _best_cost; _rwd_acc = 0; _rwd_n = 0;
                    } else if (_frozen) {                                                     // F10: commit theta<-theta_best, no perturbation, skip update
                        for (int i = 0; i < 6; i++) { _spsa_th[i] = _pol_thbest[i]; _pg_eps[i] = 0.0; _pg_thw[i] = _spsa_th[i]; }
                        _pg_phase = 0; _pg_wc = 0; _pg_wstart = _best_cost; _rwd_acc = 0; _rwd_n = 0;
                    } else {
                        double Adv, Rbar;
                        if (_pg_antithetic) { Adv = 0.5 * (_pg_Rplus - R); Rbar = 0.5 * (_pg_Rplus + R); } // F8: common-mode-rejected advantage
                        else { Adv = R - _pg_baseline; Rbar = R; }                            // current single-sided advantage
                        double _sd = std::sqrt(std::max(1e-9, _pg_r2 - _pg_baseline * _pg_baseline));
                        double A = _rwd_whiten ? Adv / _sd : Adv;                             // F7: variance-normalized advantage
                        double inv = 1.0 / (_pg_sig_w * _pg_sig_w);                           // F10: divide by the sigma that GENERATED eps (bias fix; == 1/sigma^2 when constant)
                        _pg_t++;
                        for (int i = 0; i < 6; i++) {
                            double g = A * _pg_eps[i] * inv;                                  // PGPE gradient: Adv * eps / sigma^2 (dim 5: eps[5]==0 => g==0 => theta5 stays at warm-start)
                            _pg_m[i] = 0.9 * _pg_m[i] + 0.1 * g;
                            _pg_v[i] = 0.999 * _pg_v[i] + 0.001 * g * g;
                            double mh = _pg_m[i] / (1.0 - std::pow(0.9, _pg_t));
                            double vh = _pg_v[i] / (1.0 - std::pow(0.999, _pg_t));
                            _spsa_th[i] += _pg_lr * mh / (std::sqrt(vh) + 1e-8);              // Adam ascent on windowed reward
                            if (i == 0) { if (_spsa_th[0] < _pg_w0[0]) _spsa_th[i] -= _pg_lr * _pg_lam * (_spsa_th[i] - _pg_w0[i]); } // F3: one-sided anchor (pull theta0 UP only when below warm-start)
                            if (_spsa_th[i] < -10.0) _spsa_th[i] = -10.0;
                            if (_spsa_th[i] >  10.0) _spsa_th[i] =  10.0;
                        }
                        _pg_baseline = 0.9 * _pg_baseline + 0.1 * Rbar;
                        _pg_r2       = 0.9 * _pg_r2       + 0.1 * Rbar * Rbar;                // F7: track second moment for whitening
                        double _sig_now = _pg_ssched ? (_pg_sig0 * std::pow(_pg_sigt / _pg_sig0, _p)) : _pg_sigma; // F10: annealed sigma_k (== sigma when constant)
                        _pg_sig_w = _sig_now;
                        for (int i = 0; i < 5; i++) { _pg_eps[i] = _pg_sig_w * _pg_gauss(); _pg_thw[i] = _spsa_th[i] + _pg_eps[i]; } // resample next window
                        _pg_eps[5] = 0.0; _pg_thw[5] = _spsa_th[5];                           // GAP dim: not PGPE-perturbed (no extra rand => byte-identical stream)
                        _pg_phase = 0; _pg_wc = 0; _pg_wstart = _best_cost; _rwd_acc = 0; _rwd_n = 0;
                    }
                }
            }
            else if (_accept_init == 10) {                                                    // FIX2 ablation: actor-critic with eligibility traces (soft-acceptance REINFORCE)
                double r = (cur_after < _pg_best_before)                                       // per-iter reward = relative best improvement realized THIS iter
                         ? (double)(_pg_best_before - cur_after) / (double)std::max(1, _pg_best_before) : 0.0;
                double A = r - _pg_baseline;                                                   // advantage (EMA baseline = variance reduction)
                for (int i = 0; i < 6; i++) {
                    _pg_e[i] = _pg_gamma * _pg_e[i] + _pg_step[i];                             // decay trace + add this iter's grad-log-pi
                    _pg_gbatch[i] += A * _pg_e[i];                                             // credit the improvement to recent (tolerant) decisions
                    _pg_step[i] = 0.0;                                                         // consume this iter's grad
                }
                _pg_baseline = 0.99 * _pg_baseline + 0.01 * r;
                _pg_wc++;
                if (_pg_wc >= _pg_W) {                                                         // batched Adam step (stability)
                    _pg_t++;
                    for (int i = 0; i < 6; i++) {
                        double g = _pg_gbatch[i];
                        _pg_m[i] = 0.9 * _pg_m[i] + 0.1 * g;
                        _pg_v[i] = 0.999 * _pg_v[i] + 0.001 * g * g;
                        double mh = _pg_m[i] / (1.0 - std::pow(0.9, _pg_t));
                        double vh = _pg_v[i] / (1.0 - std::pow(0.999, _pg_t));
                        _spsa_th[i] += _pg_lr * mh / (std::sqrt(vh) + 1e-8);                   // Adam ascent on trace-credited reward
                        if (i == 0) { if (_spsa_th[0] < _pg_w0[0]) _spsa_th[i] -= _pg_lr * _pg_lam * (_spsa_th[i] - _pg_w0[i]); } // F3: one-sided anchor
                        if (_spsa_th[i] < -10.0) _spsa_th[i] = -10.0;
                        if (_spsa_th[i] >  10.0) _spsa_th[i] =  10.0;
                        _pg_gbatch[i] = 0.0;
                    }
                    _pg_wc = 0;
                }
            }
        }
        runtime = ((fsec)(Time::now() - start_time)).count();
        sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;
        if (sum_of_costs < _run_best) {   // ALWAYS track best-incumbent scalar (free) -> measures return-policy gap for any mode
            _run_best = sum_of_costs;
            if (_best_return) {           // snapshot paths only when returning best (amortized negligible: only on a NEW best)
                _best_paths.resize(agents.size());
                for (size_t i = 0; i < agents.size(); i++) _best_paths[i] = agents[i].path;
            }
        }
        if (screen >= 1)
            cout << "Iteration " << iteration_stats.size() << ", "
                 << "group size = " << neighbor.agents.size() << ", "
                 << "solution cost = " << sum_of_costs << ", "
                 << "remaining time = " << time_limit - runtime << endl;
        iteration_stats.emplace_back(neighbor.agents.size(), sum_of_costs, runtime, replan_algo_name);
    }

    if (_best_return && !_best_paths.empty()) {   // restore the best incumbent so run() returns it, not the last working solution
        for (size_t i = 0; i < agents.size(); i++) agents[i].path = _best_paths[i];
        sum_of_costs = _run_best;
    }

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
         << "failed iterations = " << num_of_failures << ", "
         << "eecbs_calls = " << total_eecbs_calls << ", "
         << "eecbs_hl_expanded = " << total_eecbs_hl_expanded << ", "
         << "lb = " << sum_of_distances << ", "
         << "best_incumbent = " << _run_best << endl;   // LB (feature c) + best-so-far (return-policy gap = sum_of_costs vs best_incumbent)
    if (_accept_init == 9 || _accept_init == 10)   // log learned policy params (SPSA or policy-gradient) for mechanism analysis
        cout << "SPSA_THETA: " << _spsa_th[0] << " " << _spsa_th[1] << " " << _spsa_th[2]
             << " " << _spsa_th[3] << " " << _spsa_th[4] << " " << _spsa_th[5] << endl;
    if (_accept_init == 8)   // log bandit arm-selection counts (greedy/rr2/rr5/rr10/ta5) = which criterion it favored
        cout << "BANDIT_ARMS: " << _arm_n[0] << " " << _arm_n[1] << " " << _arm_n[2]
             << " " << _arm_n[3] << " " << _arm_n[4] << endl;
    { const char* s = std::getenv("AMOR_SEED");   // log learned DESTROY seed-policy theta (did it move from warm-start [3,0,0,0]?)
      if (s && atoi(s) == 10) cout << "SEED_THETA: " << g_sd_theta[0] << " " << g_sd_theta[1] << " "
                                   << g_sd_theta[2] << " " << g_sd_theta[3] << "  updates=" << g_sd_t << endl; }
    { const char* s = std::getenv("AMOR_REPAIR"); // log learned REPAIR-ORDER theta (did it move from [0,0]?)
      if (s && atoi(s) == 10) cout << "ORDER_THETA: " << g_ro_theta[0] << " " << g_ro_theta[1] << "  updates=" << g_ro_t << endl;
      if (s && atoi(s) == 20) cout << "REPAIR_ARMS: " << g_rb_n[0] << " " << g_rb_n[1] << " " << g_rb_n[2] << " "
                                   << g_rb_n[3] << " " << g_rb_n[4] << "  (random longest shortest most-delayed least-delayed)" << endl;
      if (s && atoi(s) == 21) { cout << "CUCB_ARMS:";
          for (int a=0;a<g_cu_K;a++) cout << " arm" << g_cu_arms[a] << "=" << g_cu_n[a]
                                          << "(mu=" << (g_cu_n[a]? g_cu_sum[a]/g_cu_n[a] : 0.0) << ")";
          cout << "  gate_open_iter=" << g_cu_gate_open << " T=" << g_cu_t
               << " mode=" << (g_cu_mode==4?"roulette":(g_cu_mode==3?"ts":(g_cu_mode==2?"budget":(g_cu_mode?"breaker":"gate")))) << " locked=" << (g_cu_locked?1:0)
               << " LOCK=" << (g_cu_lockbg?1:0) << " gated=" << (g_cu_gated?1:0)
               << " dep=" << g_cu_dep_iter
               << " That=" << (long)g_cu_That << " Nid=" << (long)g_cu_Nid << " elim=";
          for (int a=0;a<g_cu_K;a++) cout << (g_cu_elim[a]?1:0);   // audit #16: full-K elim print
          cout << endl; } }
    { const char* s = std::getenv("AMOR_SEED");  // learned LinTS/LinUCB theta [bias,log-delay,rel-delay,stretch,staleness,1/deg]
      if (s && (atoi(s)==40||atoi(s)==41)) cout << "SEED_THETA: " << g_lt_theta[0] << " " << g_lt_theta[1] << " "
             << g_lt_theta[2] << " " << g_lt_theta[3] << " " << g_lt_theta[4] << " " << g_lt_theta[5] << endl; }
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
    // CACo: symmetry-reasoning kill-switches. On some (esp. maze) neighborhood geometries the
    // rectangle/corridor/target reasoning modules carry an intermittent UB (0xC0000005) when ECBS
    // is run on a DESTROYED SUBSET (a context the original authors never exercised). They are
    // speed heuristics only — pure EECBS stays correct & bounded-suboptimal without them. Default
    // OFF the rectangle module (the confirmed culprit) so selective-EECBS never segfaults; the rest
    // stay ON unless AMOR_ECBS_SIMPLE. Env overrides let us ablate each module's quality effect.
    static const bool _ecbs_simple = (std::getenv("AMOR_ECBS_SIMPLE") != nullptr);
    static const bool _no_rect = _ecbs_simple || (std::getenv("AMOR_ECBS_RECT") == nullptr); // rect OFF by default
    static const bool _no_corr = _ecbs_simple || (std::getenv("AMOR_NO_CORR") != nullptr);
    static const bool _no_targ = _ecbs_simple || (std::getenv("AMOR_NO_TARG") != nullptr);
    ecbs.setRectangleReasoning(!_no_rect);
    ecbs.setCorridorReasoning(!_no_corr);
    ecbs.setHeuristicType(heuristics_type::WDG, heuristics_type::GLOBAL);
    ecbs.setTargetReasoning(!_no_targ);
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
    total_eecbs_calls++;
    // CACo defensive guard: ECBS can (intermittently, esp. on a root-CT timeout) report succ while
    // ecbs.paths is malformed (wrong size, or a null/empty entry). Dereferencing *ecbs.paths[i] then
    // segfaults (0xC0000005). Validate the whole vector BEFORE touching it; on any inconsistency treat
    // the call as a failure and keep the old paths (correctness-preserving, never crashes).
    bool _paths_ok = succ && ecbs.paths.size() >= neighbor.agents.size();
    if (_paths_ok)
        for (size_t i = 0; i < neighbor.agents.size(); i++)
            if (ecbs.paths[i] == nullptr || ecbs.paths[i]->empty()) { _paths_ok = false; break; }
    if (_paths_ok && ecbs.solution_cost < neighbor.old_sum_of_costs) // accept new paths
    {
        auto id = neighbor.agents.begin();
        for (size_t i = 0; i < neighbor.agents.size(); i++)
        {
            agents[*id].path = *ecbs.paths[i];
            path_table.insertPath(agents[*id].id, agents[*id].path);
            ++id;
        }
        neighbor.sum_of_costs = ecbs.solution_cost;
        total_eecbs_hl_expanded += ecbs.num_HL_expanded;
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
    auto shuffled_agents = neighbor.agents;
    // GATE-ORDER: AMOR_REPAIR switches the PP planning order. 0=random (stock), 1=longest-haul first,
    // 2=shortest-haul first (does the order matter for SOC at all? if not, the repair-order lever is a no-op).
    static const int order_mode_raw = [](){ const char* s=std::getenv("AMOR_REPAIR"); return s?atoi(s):0; }();
    static const bool order_replan_only = [](){ const char* s=std::getenv("AMOR_REPAIR_REPLANONLY"); return s&&atoi(s)!=0; }();
    // ISOLATION: if AMOR_REPAIR_REPLANONLY=1, force random order during the initial PP construction
    // (iteration_stats empty) so the init is mode-independent; the custom order only applies to anytime replan.
    // U0 hardening: bandit modes (>=20) are INTRINSICALLY replan-only — running them during the init-PP
    // construction created a phantom pull (g_cu_t++/g_cu_last set with no matching reward update in the
    // replan loop) that desynced the ledger by one. Fixed orders (1-3) still honor REPLANONLY explicitly.
    const int order_mode = ((order_replan_only || order_mode_raw >= 20) && iteration_stats.empty()) ? 0 : order_mode_raw;
    if (order_mode == 1)
        std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){
            return agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location] >
                   agents[b].path_planner->my_heuristic[agents[b].path_planner->start_location]; });
    else if (order_mode == 2)
        std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){
            return agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location] <
                   agents[b].path_planner->my_heuristic[agents[b].path_planner->start_location]; });
    else if (order_mode == 10)   // learned Plackett-Luce order (theta=0 => uniform random == stock)
    {
        ro_cfg();
        int k = (int)shuffled_agents.size();
        std::vector<std::array<double,2>> phi(k); std::vector<double> sc(k);
        double mean[2]={0,0}, m2[2]={0,0};
        for (int j=0;j<k;j++){ int id=shuffled_agents[j];
            int hd = agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location];
            int dl = (int)agents[id].path.size()-1 - hd;
            phi[j] = { (double)hd, (double)dl };
            for(int f=0;f<2;f++){ double d=phi[j][f]-mean[f]; mean[f]+=d/(j+1); m2[f]+=d*(phi[j][f]-mean[f]); } }
        double sd[2]; for(int f=0;f<2;f++){ sd[f]=(k>1)?std::sqrt(m2[f]/(k-1)):1.0; if(sd[f]<1e-6)sd[f]=1e-6; }
        for (int j=0;j<k;j++){ for(int f=0;f<2;f++) phi[j][f]=(phi[j][f]-mean[f])/sd[f];
            sc[j]=g_ro_theta[0]*phi[j][0]+g_ro_theta[1]*phi[j][1]; }
        g_ro_lb = 0; for(int id:shuffled_agents) g_ro_lb += agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location];
        g_ro_grad[0]=g_ro_grad[1]=0;
        std::vector<int> order; order.reserve(k); std::vector<char> used(k,0);
        for (int i=0;i<k;i++){                                   // Plackett-Luce: sample next ~ softmax over remaining
            double maxs=-1e18; for(int j=0;j<k;j++) if(!used[j]) maxs=std::max(maxs,sc[j]/g_ro_tau);
            double Z=0, wmean[2]={0,0};
            for(int j=0;j<k;j++) if(!used[j]){ double w=std::exp(sc[j]/g_ro_tau-maxs); Z+=w; wmean[0]+=w*phi[j][0]; wmean[1]+=w*phi[j][1]; }
            wmean[0]/=Z; wmean[1]/=Z;
            double u=((double)rand()/RAND_MAX)*Z, acc=0; int pick=-1;
            for(int j=0;j<k;j++) if(!used[j]){ acc+=std::exp(sc[j]/g_ro_tau-maxs); if(acc>=u){ pick=j; break; } }
            if(pick<0) for(int j=k-1;j>=0;j--) if(!used[j]){ pick=j; break; }
            used[pick]=1; order.push_back(shuffled_agents[pick]);
            g_ro_grad[0]+=(phi[pick][0]-wmean[0])/g_ro_tau; g_ro_grad[1]+=(phi[pick][1]-wmean[1])/g_ro_tau;
        }
        shuffled_agents = order; g_ro_have = true;
    }
    else if (order_mode == 20)   // repair-bandit: epsilon-greedy over 5 fixed order rules
    {
        rb_cfg();
        int arm = -1;
        for (int a=0;a<5;a++) if (g_rb_allow[a] && g_rb_n[a]==0) { arm=a; break; }   // warmup: each ALLOWED arm once
        if (arm<0) { if ((double)rand()/RAND_MAX < g_rb_eps) {                        // explore among allowed arms
                         int pool[5], np=0; for(int a=0;a<5;a++) if(g_rb_allow[a]) pool[np++]=a;
                         arm = pool[rand()%std::max(1,np)]; }
                     else { arm=-1; for(int a=0;a<5;a++) if(g_rb_allow[a] && (arm<0 || g_rb_val[a]>g_rb_val[arm])) arm=a; } }
        if (arm<0) arm=0;
        g_rb_last = arm; g_rb_lb = 0;
        for (int id : shuffled_agents) g_rb_lb += agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location];
        auto hd=[&](int a){ return agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]; };
        auto dd=[&](int a){ return (int)agents[a].path.size()-1 - hd(a); };
        if      (arm==0) std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
        else if (arm==1) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return hd(a)>hd(b);});
        else if (arm==2) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return hd(a)<hd(b);});
        else if (arm==3) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return dd(a)>dd(b);});
        else             std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return dd(a)<dd(b);});
    }
    else if (order_mode == 21)   // AMOR-CUCB: conservative UCB order-bandit (pre-registered; see PREREGISTRATION_cucb.md)
    {
        cu_cfg(); g_cu_t++;
        double logt = 0, logg = 0, conf0 = 0, mu0 = 0;
        // UCB-V (variance-adaptive, Audibert et al.): conf = sqrt(2*V*logt/n) + 3*logt/n.
        // Needed because the SOC-aligned slack-fraction reward has tiny mean (~0.002) — a Hoeffding
        // term sqrt(logt/2n) would drown it; the Bernstein term scales with the (equally tiny) variance.
        auto cu_conf = [&](int a)->double { if (!g_cu_n[a]) return 1.0;
            double mu = g_cu_sum[a]/g_cu_n[a];
            double v  = std::max(0.0, g_cu_sq[a]/g_cu_n[a] - mu*mu);
            return std::sqrt(2.0*v*logt/g_cu_n[a]) + 3.0*logt/g_cu_n[a]; };
        int j = 0;
        if (g_cu_mode <= 1) {   // UCB-V proposal feeds only gate/breaker; modes 2/3 assign j themselves (audit: was dead work on every BG pull)
            logt = std::log(std::max(2.0, 6.0*g_cu_K*(double)g_cu_t*(double)g_cu_t/g_cu_delta));
            j = -1; double best = -1e18;
            for (int a=0;a<g_cu_K;a++){
                double mu  = g_cu_n[a] ? g_cu_sum[a]/g_cu_n[a] : 0.0;
                double ucb = g_cu_n[a] ? mu + cu_conf(a) : 1e17;   // +inf if unexplored
                if (ucb > best){ best = ucb; j = a; } }
            logg  = std::log(std::max(2.0, 6.0*(double)g_cu_t*(double)g_cu_t/g_cu_delta));
            conf0 = cu_conf(0);
            mu0   = g_cu_n[0] ? g_cu_sum[0]/g_cu_n[0] : 0.0;
        }
        if (g_cu_mode == 3) {   // BG-TS (V3 amendment): budget-gated THOMPSON SAMPLING + anchor floor.
            // Scout-validated: SE-RR-commit's 48% capture reproduced in MC; TS+floor captures 77-79% in the
            // same regime (ETC factor-2 theorem: identify-then-commit is dominated by fully-sequential index
            // policies). Floor gives the deterministic safety share; TS gives Lai-Robbins log-regret capture.
            if (g_cu_K < 2) j = 0;
            else if ((long)g_cu_t < (long)g_cu_warm) j = 0;                   // warmup = anchor (== stock)
            else {
                if (!g_cu_gated && ((long)g_cu_t % g_cu_warm == 0)) {         // same budget gate as mode 2
                    double _elapsed  = runtime > 1e-6 ? runtime : 1e-6;
                    double _init_est = std::min((double)initial_solution_runtime, 0.5 * _elapsed);
                    double _rate = (double)g_cu_t / std::max(1e-6, _elapsed - _init_est);
                    g_cu_That = 0.8 * _rate * (time_limit - _init_est);
                    double mu0w = g_cu_n[0] ? g_cu_sum[0]/g_cu_n[0] : 0.0;
                    double varw = std::max(1e-4, (g_cu_n[0]? g_cu_sq[0]/g_cu_n[0] : 0.0) - mu0w*mu0w);
                    double lg   = std::log(std::max(2.0, 4.0*g_cu_K*g_cu_That*g_cu_That/g_cu_delta));
                    g_cu_Nid    = (g_cu_K-1) * std::ceil((2.0*varw + (2.0/3.0)*g_cu_dmin) * lg / (g_cu_dmin*g_cu_dmin));
                    g_cu_lockbg = (g_cu_Nid > g_cu_gam * (g_cu_That - (double)g_cu_warm));
                    if (!g_cu_lockbg) { g_cu_gated = true; if (g_cu_gate_open < 0) g_cu_gate_open = g_cu_t; }
                }
                if (!g_cu_gated) j = 0;
                else if (g_cu_gate_open == (long)g_cu_t) j = 0;   // certification pull itself stays anchored -> N_0(t)=t for all t<=G on every path (Thm1a), robust to W not divisible by floor period
                else if (g_cu_conf && !g_cu_dep) {   // V4 pre-departure regime (pre-reg amendment 2026-07-03)
                    if ((long)g_cu_t % 25 == 0) {    // departure test cadence (the sole new implementation parameter)
                        double A0=1.0+g_cu_sum[0], B0=1.0+std::max(0.0,(double)g_cu_n[0]-g_cu_sum[0]);
                        double m0=A0/(A0+B0), v0=A0*B0/((A0+B0)*(A0+B0)*(A0+B0+1.0));
                        for (int a=1;a<g_cu_K;a++){ if (g_cu_n[a] < 25) continue;   // minimum evidence
                            double Aa=1.0+g_cu_sum[a], Ba=1.0+std::max(0.0,(double)g_cu_n[a]-g_cu_sum[a]);
                            double ma=Aa/(Aa+Ba), va=Aa*Ba/((Aa+Ba)*(Aa+Ba)*(Aa+Ba+1.0));
                            if ((ma-m0)/std::sqrt(va+v0+1e-18) > 1.645) { g_cu_dep=true; g_cu_dep_iter=g_cu_t; break; } }
                    }
                    if (g_cu_dep) j = 0;             // departure pull itself anchored (mirrors certification-pull rule)
                    else if ((long)g_cu_t % std::max(2,(int)std::ceil(1.0/std::max(0.01,g_cu_beta))) == 0) {
                        // challenger lane at rate beta (knob reused): TS-argmax among NON-anchor arms
                        double bq=-1e18; j=1;
                        for (int a=1;a<g_cu_K;a++){ int sA=(int)std::llround(g_cu_sum[a]);
                            double q=sample_beta(1+sA, 1+std::max(0,g_cu_n[a]-sA)); if (q>bq){ bq=q; j=a; } }
                    }
                    else j = 0;                      // anchor share 1-beta pre-departure
                }
                else if ((long)g_cu_t % std::max(2,(int)std::ceil(1.0/std::max(0.01,g_cu_beta))) == 0) j = 0;  // anchor floor
                else {   // Thompson: completion channel is Bernoulli -> Beta(1+succ, 1+fail) posterior per arm
                    double bq = -1e18;
                    for (int a=0;a<g_cu_K;a++) {
                        int sA = (int)std::llround(g_cu_sum[a]);              // completion reward is 0/1 -> sum = successes
                        double q = sample_beta(1 + sA, 1 + std::max(0, g_cu_n[a]-sA));
                        if (q > bq) { bq = q; j = a; } }
                }
            }
        }
        else if (g_cu_mode == 2) {   // BG-SE (pre-registered V2): budget-gated successive elimination + anchor floor
            if (g_cu_K < 2) j = 0;                                            // degenerate arm set -> pure anchor (audit #4)
            else if ((long)g_cu_t < (long)g_cu_warm) j = 0;                   // Phase-0: warmup = pure anchor (== stock; strict < so the t=W pull triggers the first re-gate, audit #2)
            else {
                if (!g_cu_gated && ((long)g_cu_t % g_cu_warm == 0)) {         // PERIODIC re-gating (v3): early wave-contention
                                                                              // transients corrupt a one-shot projection; re-check
                                                                              // feasibility every W pulls — never-feasible == anchor (same cert).
                    // T-hat BUG FIX: project from REPLAN throughput only — total runtime includes init
                    // (0.3-1s) which halved-to-quartered the rate and over-locked the win cells.
                    double _elapsed  = runtime > 1e-6 ? runtime : 1e-6;
                    double _init_est = std::min((double)initial_solution_runtime, 0.5 * _elapsed);  // MEASURED init time (audit #7; _rt0 dead code removed)
                    double _rate = (double)g_cu_t / std::max(1e-6, _elapsed - _init_est);
                    g_cu_That = 0.8 * _rate * (time_limit - _init_est);       // conservative horizon projection (init excluded)
                    double mu0w = g_cu_n[0] ? g_cu_sum[0]/g_cu_n[0] : 0.0;
                    double varw = std::max(1e-4, (g_cu_n[0]? g_cu_sq[0]/g_cu_n[0] : 0.0) - mu0w*mu0w);
                    double lg   = std::log(std::max(2.0, 4.0*g_cu_K*g_cu_That*g_cu_That/g_cu_delta));
                    g_cu_Nid    = (g_cu_K-1) * std::ceil((2.0*varw + (2.0/3.0)*g_cu_dmin) * lg / (g_cu_dmin*g_cu_dmin));
                    g_cu_lockbg = (g_cu_Nid > g_cu_gam * (g_cu_That - (double)g_cu_warm)); // PRE-REGISTERED form: vs total-after-warmup (v4 'remaining' deviation reverted)
                    if (!g_cu_lockbg) { g_cu_gated = true; if (g_cu_gate_open < 0) g_cu_gate_open = g_cu_t; }  // audit #1: record opening
                }
                if (!g_cu_gated) j = 0;                                       // not yet certified feasible -> anchor (== stock)
                else if (g_cu_gate_open == (long)g_cu_t) j = 0;               // certification pull anchored (Thm1a robustness, mirrors mode 3)
                else if ((long)g_cu_t % std::max(2,(int)std::ceil(1.0/std::max(0.01,g_cu_beta))) == 0) j = 0;  // anchor floor at rate beta (audit #8: knob now live)
                else {                                                        // SE: eliminate provably-worse arms, round-robin survivors
                    // Elimination confidence (v6, textbook-faithful): min(Hoeffding, empirical-Bernstein) with
                    // PER-ARM log term ln(4K n^2/delta) — both valid for r in [0,1]; Bernstein's 3L/n bias term
                    // dominated at n<2000 and structurally prevented in-horizon elimination (48%-capture root cause).
                    auto se_conf = [&](int a)->double { if (!g_cu_n[a]) return 1.0;
                        double n  = (double)g_cu_n[a];
                        double L  = std::log(std::max(2.0, 10.0*g_cu_K*n*n/g_cu_delta));   // audit #9: min(H,B) union-bound constant (5 events/(arm,n) -> 10K keeps total <= delta)
                        double mu = g_cu_sum[a]/n;
                        double v  = std::max(0.0, g_cu_sq[a]/n - mu*mu);
                        double hoef = std::sqrt(L/(2.0*n));
                        double bern = std::sqrt(2.0*v*L/n) + 3.0*L/n;
                        return std::min(hoef, bern); };
                    double bestL = -1e18;
                    for (int a=0;a<g_cu_K;a++) if(!g_cu_elim[a] && g_cu_n[a])
                        bestL = std::max(bestL, g_cu_sum[a]/g_cu_n[a] - se_conf(a));
                    for (int a=1;a<g_cu_K;a++) if(!g_cu_elim[a] && g_cu_n[a]) {
                        if (g_cu_sum[a]/g_cu_n[a] + se_conf(a) < bestL) g_cu_elim[a] = true; }
                    // Round-robin over NON-ANCHOR survivors only — the beta floor already guarantees the anchor's
                    // share; giving the anchor an equal RR slot too wasted ~31% extra pulls (48%-capture cause #2).
                    int tries=0; do { g_cu_rr = 1 + (g_cu_rr % std::max(1, g_cu_K-1)); tries++; } while (g_cu_elim[g_cu_rr] && tries<=5);
                    j = g_cu_elim[g_cu_rr] ? 0 : g_cu_rr;
                }
            }
        }
        else if (g_cu_mode == 6) {   // ETC-50 (necessity row, unguarded): round-robin K*50 explore, commit to argmax mean forever
            long E = 50L * g_cu_K;
            if ((long)g_cu_t <= E) j = (int)((g_cu_t - 1) % g_cu_K);
            else { double bm = -1e18; j = 0;
                for (int a = 0; a < g_cu_K; a++){ double m = g_cu_n[a] ? g_cu_sum[a]/g_cu_n[a] : 0.0;
                    if (m > bm){ bm = m; j = a; } } }
            if (j != 0 && g_cu_gate_open < 0) g_cu_gate_open = g_cu_t;
        }
        else if (g_cu_mode == 4) {   // Ropke-Pisinger'06 roulette-wheel baseline (necessity-table row, unguarded)
            if ((long)g_cu_t > 1 && (long)g_cu_t % 100 == 1) {   // segment boundary: react weights
                for (int a=0;a<g_cu_K;a++){ if (g_cu_scn[a]) g_cu_w[a] = 0.8*g_cu_w[a] + 0.2*(g_cu_sc[a]/g_cu_scn[a]);
                    g_cu_sc[a]=0; g_cu_scn[a]=0; } }
            double tot=0; for (int a=0;a<g_cu_K;a++) tot += g_cu_w[a];
            double u = tot * ((double)rand()/((double)RAND_MAX+1.0)); j = 0;
            for (int a=0;a<g_cu_K;a++){ u -= g_cu_w[a]; if (u <= 0){ j = a; break; } }
            if (j != 0 && g_cu_gate_open < 0) g_cu_gate_open = g_cu_t;   // first non-anchor play (for logs)
        }
        else if (g_cu_mode == 1) {   // PLAN B: optimism-first UCB1 + one-way certified circuit BREAKER
            if (g_cu_locked) j = 0;   // breaker tripped earlier -> anchor for the rest of the run
            else {
                double mu0L  = std::max(0.0, mu0 - conf0);                            // LCB on baseline mean
                double slack = std::sqrt(0.5*(double)g_cu_t*logg);                    // Azuma allowance
                if (g_cu_S < (1.0-g_cu_alpha)*(double)g_cu_t*mu0L - slack) {          // pessimistic budget violation
                    g_cu_locked = true; j = 0; }
                else if (j != 0 && g_cu_gate_open < 0) g_cu_gate_open = g_cu_t;       // first non-anchor play (for logs)
            }
        }
        else if (j != 0) {   // MODE 0 SAFETY GATE: certify (1-alpha) of the anchor's cumulative reward is preserved
            double LowAlg   = g_cu_S - std::sqrt(0.5*(double)g_cu_t*logg);          // Azuma LCB on realized sum
            double muj      = g_cu_n[j] ? g_cu_sum[j]/g_cu_n[j] : 0.0;
            double LowNext  = std::max(0.0, muj - cu_conf(j));
            double HighBase = std::min(1.0, mu0 + conf0);
            if (LowAlg + LowNext < (1.0-g_cu_alpha)*(double)(g_cu_t+1)*HighBase) j = 0;   // not safe -> anchor
            else if (g_cu_gate_open < 0) g_cu_gate_open = g_cu_t;                          // first certified departure
        }
        g_cu_last = j;
        if (g_cu_reward) { g_cu_lb = 0;   // O(neighborhood) LB sum feeds only the slackfrac reward (audit: dead work under completion)
            for (int id : shuffled_agents) g_cu_lb += agents[id].path_planner->my_heuristic[agents[id].path_planner->start_location]; }
        int arm = g_cu_arms[j];
        auto hd=[&](int a){ return agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]; };
        auto dd=[&](int a){ return (int)agents[a].path.size()-1 - hd(a); };
        if      (arm==0) std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
        else if (arm==1) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return hd(a)>hd(b);});
        else if (arm==2) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return hd(a)<hd(b);});
        else if (arm==3) std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return dd(a)>dd(b);});
        else             std::sort(shuffled_agents.begin(), shuffled_agents.end(), [&](int a,int b){return dd(a)<dd(b);});
    }
    else
        std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());
    // AMOR_ACCEPT (zero-overhead non-greedy acceptance): in non-greedy modes PP must return a COMPLETE
    // neighborhood solution (even if costlier than old) so run() can apply the acceptance criterion.
    // Skip the greedy early-break/cost-revert; still revert on planning FAILURE. Only active during replan.
    static const int _accept_mode = [](){ const char* s = std::getenv("AMOR_ACCEPT");
        if (!s) return 0; std::string v = s; if (v=="lahc") return 1; if (v=="rr") return 2; if (v=="reheat") return 3; if (v=="ta") return 4; if (v=="adapt") return 5; if (v=="adaptc") return 6; if (v=="adaptsw") return 7; if (v=="bandit") return 8; if (v=="spsa") return 9; if (v=="pg") return 10; if (v=="gd") return 11; return 0; }();   // gd was MISSING here -> PP kept its greedy early-break under AMOR_ACCEPT=gd, silently making great-deluge inert (the gd "DEAD" verdict was partly this bug)
    bool nongreedy = (_accept_mode != 0) && !iteration_stats.empty();
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
        if (!nongreedy && neighbor.sum_of_costs >= neighbor.old_sum_of_costs)
            break;
        remaining_agents--;
        path_table.insertPath(agents[id].id, agents[id].path);
        ++p;
    }
    if (remaining_agents == 0 && (nongreedy || neighbor.sum_of_costs <= neighbor.old_sum_of_costs)) // keep complete repair (accept decided in run())
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
//    solver.WarshallFloyd();
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

//    Graph* G = new SimpleGrid(instance);
    Graph* G = new SimpleGrid(instance.getMapFile());

    std::vector<Task*> T;
    PIBT_Agents A;

    for (int i : shuffled_agents){
        assert(G->existNode(agents[i].path_planner->start_location));
        assert(G->existNode(agents[i].path_planner->goal_location));
        auto a = new PIBT_Agent(G->getNode( agents[i].path_planner->start_location));

//        PIBT_Agent* a = new PIBT_Agent(G->getNode( agents[i].path_planner.start_location));
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
    rouletteWheel();
    switch (selected_neighbor)
    {
        case 0 : destroy_strategy = RANDOMWALK; break;
        case 1 : destroy_strategy = INTERSECTION; break;
        case 2 : destroy_strategy = RANDOMAGENTS; break;
        default : cerr << "ERROR" << endl; exit(-1);
    }
}

bool LNS::generateNeighborByIntersection()
{
    if (intersections.empty())
    {
        for (int i = 0; i < instance.map_size; i++)
        {
            if (!instance.isObstacle(i) && instance.getDegree(i) > 2)
                intersections.push_back(i);
        }
    }

    set<int> neighbors_set;
    auto pt = intersections.begin();
    std::advance(pt, rand() % intersections.size());
    int location = *pt;
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
bool LNS::generateNeighborByRandomWalk()
{
    if (neighbor_size >= (int)agents.size())
    {
        neighbor.agents.resize(agents.size());
        for (int i = 0; i < (int)agents.size(); i++)
            neighbor.agents[i] = i;
        return true;
    }

    int a = findMostDelayedAgent();
    if (a < 0)
        return false;
    
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
                a = n;
                break;
            }
            i++;
        }
    }
    if (neighbors_set.size() < 2)
        return false;
    neighbor.agents.assign(neighbors_set.begin(), neighbors_set.end());
    if (screen >= 2)
        cout << "Generate " << neighbor.agents.size() << " neighbors by random walks of agent " << a
             << "(" << agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location]
             << "->" << agents[a].path.size() - 1 << ")" << endl;

    return true;
}

int LNS::findMostDelayedAgent()
{
    // GATE-DESTROY: AMOR_SEED switches the seed-selection rule. 0=argmax+tabu (stock), 1=prop-delay,
    // 2=uniform-delayed, 3=argmax-no-tabu, 10=SMOOTH learned policy, 20=ADDRESS-style Thompson bandit.
    static const int seed_mode = [](){ const char* s = std::getenv("AMOR_SEED"); return s ? atoi(s) : 0; }();
    const int n = (int)agents.size();
    if (seed_mode == 10)   // smooth feature policy among the SAME hard-tabu-filtered delayed agents as stock
    {
        sd_cfg();
        if ((int)g_sd_yema.size() != n) { g_sd_yema.assign(n, 0.0); g_sd_touched.assign(n, 0); }
        static std::vector<int> cand; static std::vector<std::array<double,4>> phi;
        cand.clear(); phi.clear();
        double mean[4]={0,0,0,0}, m2[4]={0,0,0,0}; int c=0;
        for (int i=0;i<n;i++){ int d=agents[i].getNumOfDelays(); if(d<=0) continue;
            if (tabu_list.find(i) != tabu_list.end()) continue;   // HARD tabu, exactly like stock (diversity == stock)
            int h=agents[i].path_planner->my_heuristic[agents[i].path_planner->start_location];
            std::array<double,4> f = { std::log1p((double)d), (double)d/(double)(h+1),
                                       std::log1p((double)(g_sd_iter-g_sd_touched[i])), g_sd_yema[i] };
            cand.push_back(i); phi.push_back(f);
            c++; for(int k=0;k<4;k++){ double dd=f[k]-mean[k]; mean[k]+=dd/c; m2[k]+=dd*(f[k]-mean[k]); } }
        if (cand.empty()) { tabu_list.clear(); return -1; }       // all high-delay tabu'd -> clear (like stock)
        double sd[4]; for(int k=0;k<4;k++){ sd[k]=(c>1)?std::sqrt(m2[k]/(c-1)):1.0; if(sd[k]<1e-6)sd[k]=1e-6; }
        static std::vector<std::array<double,4>> zs; static std::vector<double> sc; zs.clear(); sc.clear();
        int best=-1; double bestval=-1e18, bz[4]={0,0,0,0}, maxsc=-1e18;
        for (size_t j=0;j<cand.size();j++){
            std::array<double,4> z; double score=0; for(int k=0;k<4;k++){ z[k]=(phi[j][k]-mean[k])/sd[k]; score+=g_sd_theta[k]*z[k]; }
            zs.push_back(z); sc.push_back(score); if(score>maxsc) maxsc=score;
            double u=((double)rand()+1.0)/((double)RAND_MAX+2.0);
            double val=score/g_sd_tau + (-std::log(-std::log(u)));                 // small tau => near-deterministic argmax(score); frozen delay-only == stock
            if(val>bestval){ bestval=val; best=cand[j]; for(int k=0;k<4;k++) bz[k]=z[k]; } }
        double wsum=0, pmean[4]={0,0,0,0};                                          // exact softmax policy-mean feature = E_pi[phi] (unbiased REINFORCE baseline)
        for (size_t j=0;j<sc.size();j++){ double w=std::exp((sc[j]-maxsc)/g_sd_tau); wsum+=w; for(int k=0;k<4;k++) pmean[k]+=w*zs[j][k]; }
        for(int k=0;k<4;k++){ g_sd_phi_chosen[k]=bz[k]; g_sd_phibar[k]=(wsum>0)?pmean[k]/wsum:bz[k]; }
        tabu_list.insert(best); if ((int)tabu_list.size()==n) tabu_list.clear();   // maintain tabu, exactly like stock
        g_sd_last=best; return best;
    }
    if (seed_mode == 20)   // ADDRESS-style Thompson bandit over top-K most-delayed
    {
        ts_cfg();
        if ((int)g_ts_a.size() != n) { g_ts_a.assign(n,1.0); g_ts_b.assign(n,1.0); }
        static std::vector<std::pair<int,int>> dl; dl.clear();
        for (int i=0;i<n;i++){ int d=agents[i].getNumOfDelays(); if(d>0) dl.push_back({d,i}); }
        if (dl.empty()) return -1;
        int K=std::min(g_ts_K,(int)dl.size());
        std::partial_sort(dl.begin(), dl.begin()+K, dl.end(), [](const std::pair<int,int>&x,const std::pair<int,int>&y){return x.first>y.first;});
        int best=-1; double bestq=-1e18;
        for (int j=0;j<K;j++){ int id=dl[j].second;
            double q = sample_beta((int)g_ts_a[id], (int)g_ts_b[id]);   // B2 FIX: TRUE Beta Thompson (ADDRESS Alg.1 l.5-7), not a Gaussian moment-match
            if(q>bestq){ bestq=q; best=id; } }
        g_ts_last=best; return best;
    }
    if (seed_mode == 30)   // TACKLE MABUC: roulette intent -> Thompson over the intent-row of the K x K Beta table
    {
        ts_cfg(); int K = g_ts_K;
        if ((int)g_tk_alpha.size()!=K){ g_tk_alpha.assign(K, std::vector<int>(K,1)); g_tk_beta.assign(K, std::vector<int>(K,1)); }
        static std::vector<std::pair<int,int>> dl; dl.clear(); long total=0;
        for (int i=0;i<n;i++){ int d=agents[i].getNumOfDelays(); if(d>0){ dl.push_back({d,i}); total+=d; } }
        if (dl.empty()) return -1;
        int KK=std::min(K,(int)dl.size());
        std::partial_sort(dl.begin(), dl.begin()+KK, dl.end(), [](const std::pair<int,int>&x,const std::pair<int,int>&y){return x.first>y.first;});
        long rr=(long)(((double)rand()/RAND_MAX)*total), acc=0; int intent_idx=0;   // roulette intent over top-K delays
        for (int j=0;j<KK;j++){ acc+=dl[j].first; if(acc>=rr){ intent_idx=j; break; } }
        int best=-1; double bestq=-1.0;                                              // Thompson over the intent row
        for (int j=0;j<KK;j++){ double q=sample_beta(g_tk_alpha[intent_idx][j], g_tk_beta[intent_idx][j]); if(q>bestq){ bestq=q; best=j; } }
        g_tk_intent=intent_idx; g_tk_col=best; return dl[best].second;
    }
    if (seed_mode == 39)   // PROBE: uniformly-random seed from top-K delayed non-tabu (unbiased log for offline fitting)
    {
        if ((int)g_pr_touched.size() != n) g_pr_touched.assign(n, 0);
        static std::vector<std::pair<int,int>> dl; dl.clear();
        for (int i=0;i<n;i++){ int d=agents[i].getNumOfDelays(); if(d>0 && tabu_list.find(i)==tabu_list.end()) dl.push_back({d,i}); }
        if (dl.empty()) { tabu_list.clear(); return -1; }
        int K=std::min(32,(int)dl.size());
        std::partial_sort(dl.begin(), dl.begin()+K, dl.end(), [](const std::pair<int,int>&x,const std::pair<int,int>&y){return x.first>y.first;});
        int pick=rand()%K, a=dl[pick].second, d=dl[pick].first;
        int h=agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location];
        int L=(int)agents[a].path.size(), cc=0, cn=0, step=std::max(1, L/24);   // corridor_frac (subsampled)
        for (int j=0;j<L;j+=step){ cn++; if(instance.getDegree(agents[a].path[j].location)<=2) cc++; }
        g_pr_phi[0]=std::log1p((double)d);                     // log-delay
        g_pr_phi[1]=(double)d/(double)(h+1);                   // relative delay
        g_pr_phi[2]=(double)(L-1)/(double)(h+1);               // path-stretch
        g_pr_phi[3]=std::log1p((double)(g_pr_iter-g_pr_touched[a])); // staleness
        g_pr_phi[4]=1.0/(double)std::max(1,instance.getDegree(agents[a].path_planner->start_location)); // bottleneck 1/deg
        g_pr_phi[5]=(cn>0)?(double)cc/(double)cn:0.0;          // corridor_frac (topological congestion)
        g_pr_touched[a]=(int)g_pr_iter; g_pr_iter++; g_pr_last=a;
        tabu_list.insert(a); if ((int)tabu_list.size()==n) tabu_list.clear();
        return a;
    }
    if (seed_mode == 40 || seed_mode == 41)   // CONTEXTUAL LINEAR bandit (40=Thompson, 41=UCB) over 6 O(1) features
    {
        lt_cfg(); lt_setup();
        if ((int)g_lt_touched.size() != n) g_lt_touched.assign(n, 0);
        static std::vector<std::pair<int,int>> dl; dl.clear();
        for (int i=0;i<n;i++){ int d=agents[i].getNumOfDelays(); if(d>0 && tabu_list.find(i)==tabu_list.end()) dl.push_back({d,i}); }
        if (dl.empty()) { tabu_list.clear(); return -1; }
        int K=std::min(g_lt_K,(int)dl.size());
        std::partial_sort(dl.begin(), dl.begin()+K, dl.end(), [](const std::pair<int,int>&x,const std::pair<int,int>&y){return x.first>y.first;});
        static std::vector<std::array<double,6>> raw; raw.clear();
        for (int j=0;j<K;j++){ int a=dl[j].second, d=dl[j].first;
            int h=agents[a].path_planner->my_heuristic[agents[a].path_planner->start_location];
            int plen=(int)agents[a].path.size()-1, deg=instance.getDegree(agents[a].path_planner->start_location);
            raw.push_back({1.0, std::log1p((double)d), (double)d/(double)(h+1), (double)plen/(double)(h+1),
                           std::log1p((double)(g_lt_iter-g_lt_touched[a])), 1.0/(double)std::max(1,deg)}); }
        for (int j=0;j<K;j++) for(int k=1;k<6;k++){ g_lt_mu[k]=0.999*g_lt_mu[k]+0.001*raw[j][k];
            double dv=raw[j][k]-g_lt_mu[k]; g_lt_var[k]=0.999*g_lt_var[k]+0.001*dv*dv; }
        double tt[6]; for(int k=0;k<6;k++) tt[k]=g_lt_theta[k] + ((seed_mode==40)? g_lt_v*std::sqrt(std::max(0.0,g_lt_Ainv[k][k]))*sd_gauss() : 0.0);
        int best=-1; double bestsc=-1e18, bestphi[6]={0};
        for (int j=0;j<K;j++){ double phi[6], sc=0, quad=0;
            for(int k=0;k<6;k++){ phi[k]=(k==0)?1.0:(raw[j][k]-g_lt_mu[k])/std::sqrt(std::max(1e-9,g_lt_var[k]));
                sc += ((seed_mode==40)?tt[k]:g_lt_theta[k])*phi[k]; if(seed_mode==41) quad+=phi[k]*phi[k]*g_lt_Ainv[k][k]; }
            if(seed_mode==41) sc += g_lt_alpha*std::sqrt(std::max(0.0,quad));
            if(sc>bestsc){ bestsc=sc; best=dl[j].second; for(int k=0;k<6;k++) bestphi[k]=phi[k]; } }
        for(int k=0;k<6;k++) g_lt_xlast[k]=bestphi[k];
        g_lt_touched[best]=(int)g_lt_iter; g_lt_iter++; g_lt_last=best;
        tabu_list.insert(best); if ((int)tabu_list.size()==n) tabu_list.clear();
        return best;
    }
    if (seed_mode == 1 || seed_mode == 2)
    {
        long total = 0; int cnt = 0;
        for (int i = 0; i < n; i++) { int d = agents[i].getNumOfDelays(); if (d > 0) { total += d; cnt++; } }
        if (cnt == 0) return -1;
        if (seed_mode == 2) { int pick = rand() % cnt, j = 0;
            for (int i = 0; i < n; i++) if (agents[i].getNumOfDelays() > 0) { if (j == pick) return i; j++; } }
        else { long pt = rand() % total + 1, sum = 0;
            for (int i = 0; i < n; i++) { int d = agents[i].getNumOfDelays(); if (d > 0) { sum += d; if (sum >= pt) return i; } } }
        return -1;
    }
    int a = -1;
    int max_delays = -1;
    for (int i = 0; i < agents.size(); i++)
    {
        if (seed_mode != 3 && tabu_list.find(i) != tabu_list.end())
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
    if (seed_mode != 3)
    {
        tabu_list.insert(a);
        if (tabu_list.size() == agents.size())
            tabu_list.clear();
    }
    return a;
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
    output << "num of agents," <<
           "sum of costs," <<
           "runtime," <<
           "cost lowerbound," <<
           "sum of distances," <<
           "MAPF algorithm" << endl;

    // FF-12 / rereview C4: event-driven output — write only rows where the RUNNING-MIN incumbent
    // improved (plus the first row). Full per-iteration dumps were 2-6 MB/run (100-400 GB per
    // battery); improvement events are 10^2-10^3 rows/run. Output-side filter only — the in-memory
    // stats and all solver behavior are unchanged. The running-min also fixes PI(T) under
    // non-greedy acceptance (logged cost = accepted, not incumbent-best).
    int best_so_far = INT_MAX; bool first = true;
    for (const auto &data : iteration_stats)
    {
        if (!first && data.sum_of_costs >= best_so_far) continue;
        first = false; best_so_far = min(best_so_far, data.sum_of_costs);
        output << data.num_of_agents << "," <<
               data.sum_of_costs << "," <<
               data.runtime << "," <<
               max(sum_of_costs_lowerbound, sum_of_distances) << "," <<
               sum_of_distances << "," <<
               data.algorithm << endl;
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
                 "preprocessing runtime,solver name,instance name" << endl;
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
          max(sum_of_distances, sum_of_costs_lowerbound) << "," << sum_of_distances << "," <<
          iteration_stats.size() << "," << average_group_size << "," <<
          initial_solution_runtime << "," << restart_times << "," << auc << "," <<
          num_LL_expanded << "," << num_LL_generated << "," << num_LL_reopened << "," << num_LL_runs << "," <<
          preprocessing_time << "," << getSolverName() << "," << instance.getInstanceName() << endl;
    stats.close();
}

void LNS::writePathsToFile(const string & file_name) const
{
    std::ofstream output;
    output.open(file_name);
    // header
    // output << agents.size() << endl;

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
