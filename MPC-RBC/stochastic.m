function mpc_benchmark()
%% Benchmark: Deterministic MPC vs Scenario-based Stochastic MPC
%  - Plant: stormwater pond (pondcstr_StateFcn / pondcstr_OutputFcn)
%  - Disturbances:
%       MD_t = [qin_t, cin_t]     (true)
%       MD   = [qin_f, cin_f]     (forecast / nominal)
%  - Controllers:
%       1) Deterministic MPC (DMPC) via nlmpc
%       2) Scenario-based Stochastic MPC (SMPC, Monte Carlo)

clearvars; close all; clc;

%% ------------------------------------------------------------------------
% 1. Data preparation
% -------------------------------------------------------------------------

data1 = readtable('pond4_background.csv');  % true inflow/concentration
data2 = readtable('pond4_background.csv');  % forecast inflow/concentration


extra_len = 789;
qin_t = [data1.qin; zeros(extra_len,1)];  % true inflow
cin_t = [data1.cin; zeros(extra_len,1)];  % true concentration

qin_f = [data2.qin; zeros(extra_len,1)];  % forecast inflow

EMC   = sum(data1.qin .* data1.cin) / sum(data1.qin);
qin_len = length(qin_t);
cin_f = EMC * ones(qin_len,1);

MD_t = [qin_t, cin_t];   % true disturbances
MD   = [qin_f, cin_f];   % forecast disturbances

Tsim = length(qin_t);    % Total simulation time

%% ------------------------------------------------------------------------
% 2. Nonlinear MPC Design (same as deterministic version)
% -------------------------------------------------------------------------

% x = [h, c] (2 states)
% y = [h, c, q_out] (3 outputs)
% MV:   theta (orifice opening ratio)
% MD:   [qin, cin]
nlmpcobj = nlmpc(2, 3, 'MV', 1, 'MD', [2,3]);

% MPC sample time (dimensionless index); physical dt is 15 min in state equations
Ts = 1;
nlmpcobj.Ts = Ts;

% temprary horizons
nlmpcobj.PredictionHorizon = Tsim;
nlmpcobj.ControlHorizon    = 2;

% 
nlmpcobj.Model.StateFcn        = @(x,u) pondcstr_StateFcn(x, u);
nlmpcobj.Model.OutputFcn       = @(x,u) pondcstr_OutputFcn(x, u);
nlmpcobj.Model.IsContinuousTime = false;

x0 = [0.01, 0];  % [h0, c0] in ft / concentration units
u0 = 1;          % valve fully open

nlmpcobj.Optimization.CustomCostFcn       = 'pondcstrCostFcn';
nlmpcobj.Optimization.ReplaceStandardCost = true;

yref = [0 0 0];  

%% ------------------------------------------------------------------------
% 3. Passive system: valve always open (baseline)
% -------------------------------------------------------------------------

nlmpcobj.MV(1).Min = 1;
nlmpcobj.MV(1).Max = 1;

fprintf('\nPassive system simulation...\n');
[~,~,nc] = nlmpcmove(nlmpcobj, x0, u0, [], MD_t);  % open-loop passive traj

%% ------------------------------------------------------------------------
% 4. Deterministic MPC (DMPC) closed-loop with forecast MD
%    - Controller sees forecast MD
%    - True plant evolves under true MD_t
% -------------------------------------------------------------------------

horizonDet = 96;          % 24 hours (96 * 15min)
nlmpcobj.PredictionHorizon = horizonDet;
nlmpcobj.ControlHorizon    = 2;

% MV bounds for normal operation
nlmpcobj.MV(1).Min = 0;
nlmpcobj.MV(1).Max = 1;

% Pond height constraint (ft)
hlimit_ft = 10;
nlmpcobj.State(1).Max = hlimit_ft;

% Preallocate trajectories
x_det  = zeros(Tsim, 2);  % [h, c]
y_det  = zeros(Tsim, 1);  % q_out
u_det  = zeros(Tsim, 1);  % theta

x_det(1,:) = x0;
u_det(1)   = u0;
y_det(1)   = 0;

fprintf('\nDeterministic MPC closed-loop simulation...\n');
tic;
for k = 1:(Tsim - horizonDet)
    xk = x_det(k,:);      % the current time step
    uk_prev = u_det(k);   % previous control input

    % the nominal disturbance over the horizon
    MD_nom = MD(k:(k + horizonDet - 1), :);

    % Solve NMPC optimization
    [u_opt, ~, ~] = nlmpcmove(nlmpcobj, xk, uk_prev, yref, MD_nom);

    % u_opt is optimal control at time k
    u_k = u_opt;  % scalar
    u_det(k,1) = u_k;

    % Apply to true plant with true disturbance
    u_real = [u_k, MD_t(k,:)];               % [theta, qin_true, cin_true]
    x_next = pondcstr_StateFcn(xk, u_real);
    y_next = pondcstr_OutputFcn(xk, u_real);

    x_det(k+1,:) = x_next';                  % save next state
    y_det(k+1,1) = y_next(3);                % save true outflow
end
timeDet = toc;
fprintf('Deterministic MPC finished. Elapsed time = %.2f s\n', timeDet);

%% ------------------------------------------------------------------------
% 5. Scenario-based Stochastic MPC (SMPC) benchmark
% -------------------------------------------------------------------------
%  - Uncertainty: forecast errors in q_in and c_in
%  - Objective: minimize expected cost E[J']
%  - Chance constraint: Pr(h <= h_max) >= 1 - epsilon

% SMPC parameters
Ns      = 10;                         % number of scenarios
H_smpc  = 96;                         % SMPC prediction horizon (steps)
theta_candidates = linspace(0,1,11);  % candidate valve openings
epsilon = 0.05;                       % allowed overflow risk (5%)

% Relative disturbance intensity (tunable)
sigma_q = 0.30;   % inflow prediction std ~ 30%
sigma_c = 0.30;   % concentration prediction std ~ 30%

% Preallocate SMPC trajectories
sth  = zeros(Tsim,1);   % SMPC real height
stc  = zeros(Tsim,1);   % SMPC real concentration
sty  = zeros(Tsim,1);   % SMPC real outflow
stmv = zeros(Tsim,1);   % SMPC control

sth(1)  = x0(1);
stc(1)  = x0(2);
sty(1)  = 0;
stmv(1) = u0;

fprintf('\nStochastic MPC (scenario-based) simulation...\n');
tic;
for k = 1:(Tsim - H_smpc)
    % ---------- 1. Nominal prediction ----------
    q_base = qin_f(k : k+H_smpc-1);   % forecast inflow
    c_base = cin_f(k : k+H_smpc-1);   % forecast concentration

    % ---------- 2. Scenario sampling ----------
    % q_s = q_base * (1 + sigma_q * N(0,1))
    q_s = repmat(q_base,1,Ns) .* (1 + sigma_q * randn(H_smpc,Ns));
    c_s = repmat(c_base,1,Ns) .* (1 + sigma_c * randn(H_smpc,Ns));

    % keep non-negative
    q_s = max(0, q_s);
    c_s = max(0, c_s);

    % ---------- 3. Evaluate each candidate theta ----------
    best_cost  = Inf;
    best_theta = stmv(k);  % default: keep last control

    for th = theta_candidates
        J_vals      = zeros(Ns,1);
        violate_cnt = 0;

        for s = 1:Ns
            h_s    = sth(k);
            c_scen = stc(k);
            J_s    = 0;

            for j = 1:H_smpc
                q_in_s = q_s(j,s);
                c_in_s = c_s(j,s);
                u_s    = [th, q_in_s, c_in_s];

                x_next_s = pondcstr_StateFcn([h_s, c_scen], u_s);
                y_next_s = pondcstr_OutputFcn([h_s, c_scen], u_s);

                q_out_s = y_next_s(3);

                % stage cost: simplified pondcstrCostFcn
                J_stage = 5*(c_scen * q_out_s)^2 + ...
                          (q_out_s - mean(q_base))^2;
                J_s = J_s + J_stage;

                h_s    = x_next_s(1);
                c_scen = x_next_s(2);

                % overflow count for chance constraint
                if h_s > hlimit_ft
                    violate_cnt = violate_cnt + 1;
                end
            end

            % terminal penalty on height
            J_s = J_s + 900*(h_s^2);
            J_vals(s) = J_s;
        end

        prob_violate = violate_cnt / (Ns * H_smpc);

        % chance constraint: Pr(h <= h_max) >= 1 - epsilon
        if prob_violate > epsilon
            continue;
        end

        % expected cost
        J_exp = mean(J_vals);
        if J_exp < best_cost
            best_cost  = J_exp;
            best_theta = th;
        end
    end

    % ---------- 4. Apply best_theta to true plant ----------
    u_real = [best_theta, MD_t(k,:)];      % use true disturbances
    x_next = pondcstr_StateFcn([sth(k), stc(k)], u_real);
    y_next = pondcstr_OutputFcn([sth(k), stc(k)], u_real);

    sth(k+1)  = x_next(1);
    stc(k+1)  = x_next(2);
    sty(k+1)  = y_next(3);
    stmv(k+1) = best_theta;
end
timeSMPC = toc;
fprintf('Stochastic MPC finished. Elapsed time = %.2f s\n', timeSMPC);

%% ------------------------------------------------------------------------
% 6. Unit conversion (US -> SI) & Performance metrics
% -------------------------------------------------------------------------

% unit conversion
ft2_to_m2   = 1/10.764;
ft_to_m     = 1/3.281;
cfs_to_m3s  = 0.028316846592;
Amax_ft2    = 134285;            %
Amax_m2     = Amax_ft2 * ft2_to_m2;

% 真实 inflow (for info)
qin_t_m3s = qin_t * cfs_to_m3s;

% Passive system (open valve) trajectories（来自 nc）
h_pas_m  = nc.Xopt(:,1) * ft_to_m;
c_pas    = nc.Xopt(:,2);
q_pas_m3s = nc.Yopt(:,3) * cfs_to_m3s;

% DMPC trajectories
h_det_m  = x_det(:,1) * ft_to_m;
c_det    = x_det(:,2);
q_det_m3s = y_det * cfs_to_m3s;

% SMPC trajectories
h_smpc_m  = sth * ft_to_m;
c_smpc    = stc;
q_smpc_m3s = sty * cfs_to_m3s;

% 
T_eval = Tsim - max(horizonDet, H_smpc);
idx    = 1:T_eval;

% （10 ft -> m）
h_over_m = 10 * ft_to_m;

% ---- Overflow volume (m^3) ----
overflow_det  = max(0, (max(h_det_m(idx))  - h_over_m) * Amax_m2);
overflow_smpc = max(0, (max(h_smpc_m(idx)) - h_over_m) * Amax_m2);
overflow_pas  = max(0, (max(h_pas_m(idx))  - h_over_m) * Amax_m2);

overflow_vec = [overflow_det; overflow_smpc; overflow_pas];
overflow_pct = overflow_vec / max(overflow_vec) * 100;

% ---- Peak outflow (m^3/s) ----
peak_det  = max(q_det_m3s(idx));
peak_smpc = max(q_smpc_m3s(idx));
peak_pas  = max(q_pas_m3s(idx));

peak_vec = [peak_det; peak_smpc; peak_pas];
peak_pct = peak_vec / max(peak_vec) * 100;

% ---- Cumulative pollutant load (arbitrary units, e.g. kg) ----
dt = 15 * 60;   % 15 min in seconds

load_det  = sum(c_det(idx)   .* q_det_m3s(idx))   * 1e-3 * dt;
load_smpc = sum(c_smpc(idx)  .* q_smpc_m3s(idx))  * 1e-3 * dt;
load_pas  = sum(c_pas(idx)   .* q_pas_m3s(idx))   * 1e-3 * dt;

load_vec = [load_det; load_smpc; load_pas];
load_pct = load_vec / max(load_vec) * 100;

% ---- Control effort (sum of squared changes in theta) ----
coneff_det  = sum(diff(u_det(idx)).^2);
coneff_smpc = sum(diff(stmv(idx)).^2);
% Passive 
coneff_pas  = 0;

coneff_vec = [coneff_det; coneff_smpc; coneff_pas];
coneff_pct = coneff_vec / max(max(coneff_vec),eps) * 100;

% ---- Outflow smoothness (variance-like) ----
smooth_det  = sum((q_det_m3s(idx)  - mean(q_det_m3s(idx))).^2);
smooth_smpc = sum((q_smpc_m3s(idx) - mean(q_smpc_m3s(idx))).^2);
smooth_pas  = sum((q_pas_m3s(idx)  - mean(q_pas_m3s(idx))).^2);

smooth_vec = [smooth_det; smooth_smpc; smooth_pas];
smooth_pct = smooth_vec / max(smooth_vec) * 100;

%% ------------------------------------------------------------------------
% 7. Print and save results
% -------------------------------------------------------------------------

controller_names = {'Deterministic MPC'; 'Stochastic MPC'; 'Passive (valve=1)'};

fprintf('\n=== Performance comparison (US->SI converted) ===\n');
fprintf('Controller              Overflow(m^3)  PeakQ(m^3/s)  Load(arb)   Effort    Smooth\n');
for i = 1:3
    fprintf('%-20s  %10.2f   %10.3f   %10.2f   %7.3f   %10.2f\n', ...
        controller_names{i}, ...
        overflow_vec(i), peak_vec(i), load_vec(i), coneff_vec(i), smooth_vec(i));
end

save('mpc_benchmark_results.mat', ...
    'x_det','y_det','u_det', ...
    'sth','stc','sty','stmv', ...
    'nc', ...
    'overflow_vec','overflow_pct', ...
    'peak_vec','peak_pct', ...
    'load_vec','load_pct', ...
    'coneff_vec','coneff_pct', ...
    'smooth_vec','smooth_pct', ...
    'qin_t','cin_t','qin_f','cin_f');

fprintf('\nResults saved to mpc_benchmark_results.mat\n');

end % end of main function

%% ========================================================================
% Local functions: pondcstr_OutputFcn, pondcstr_StateFcn, pondcstrCostFcn
% ========================================================================

% same as original file
function y = pondcstr_OutputFcn(x, u)
%  y = [h; c; q_out]

% States
h = x(1);
c = x(2);

% Inputs
theta = u(1);
q_in  = u(2); %#ok<NASGU>
c_in  = u(3); %#ok<NASGU>

% Parameters
co = 0.65;
Ao = 1;
g  = 32.2;

y       = zeros(3,1);
y(1)    = h;
y(2)    = c;
y(3)    = theta * co * Ao * sqrt(2*g*h) * min(1,h);  % outflow

end

function x_dt = pondcstr_StateFcn(x, u)
% (discrete time, 15 min step)
% x = [h; c]
% u = [theta, q_in, c_in]

h     = x(1);
c     = x(2);
theta = u(1);
q_in  = u(2);
c_in  = u(3);

x_dt = zeros(2,1);

% Parameters
co = 0.65;
Ao = 1;
g  = 32.2;
k  = 0.8/24/60/60;   % 1/s
dt = 15*60;          % 15 minutes

% Pond geometry (node 4, from original code)
h = max(0, h);
elevation = [0, 2, 4, 6, 8, 10];
area      = [82971, 93258, 106100, 119152, 134285, 134285];
A         = interp1(elevation, area, h,'spline');

% Outflow
q_out = theta * co * Ao * sqrt(2*g*h) * min(1,h);

% Water balance
x_dt(1) = max(0, h + dt/A * (q_in - q_out));

% Pollutant mass balance with first-order decay
if h > 0
    x_dt(2) = (c*A*h*exp(-k*dt) + c_in*q_in*dt) / (A*h + q_in*dt);
else
    x_dt(2) = 0;
end

end

function f = pondcstrCostFcn(X, U, e, data) %#ok<INUSL>

p  = data.PredictionHorizon;
U1 = U(1:p, data.MVIndex(1));  % theta over horizon
X1 = X(2:p+1,1);               % h over horizon
X2 = X(2:p+1,2);               % c over horizon

co = 0.65;
Ao = 1;
g  = 32.2;

q_out = U1 .* co .* Ao .* sqrt(2*g.*X1) .* min(1,X1);


term_load = 5 * sum((X2 .* q_out).^2);


q_mean = (1/p) * sum(q_out);
term_smooth = sum((q_out - q_mean).^2);


term_terminal = 900 * (X1(end).^2);

f = term_load + term_smooth + term_terminal;

end
