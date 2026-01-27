total_agents = 1;
n_start = 4;

beta = 3;

global e_factors;

d_min = 1;
alpha = 3;
x_min = 1;
b_min = 0.5;

total_simulations=60;
options = optimoptions("fmincon","MaxIterations",1e5,"MaxFunctionEvaluations",1e5,"EnableFeasibilityMode",true,"SubproblemAlgorithm","cg");  

agent_count = 1;

share_discretization = 10; %% The number of divisions of the [0,1] interval
share_list = linspace(0.01,1,share_discretization); %% obtain a discretization of the [0,1] interval. The principal will optimize over what share to give agents

hill_climb_tolerance = 0.01;
bhill_climb_tolerance = 0.01;
avg_prod_comparison = zeros(total_agents,1);
avg_b_comparison = zeros(total_agents,1);
avg_b_comparison_fm = zeros(total_agents,1);

%max_sigmoid_step = 6; %% These are for a sigmoid step function
%min_sigmoid_step = 3; %% These are for a sigmoid step function
%total_sigmoid_step = 15;
%sigmoid_step = linspace(min_sigmoid_step,max_sigmoid_step,total_sigmoid_step);

max_exp_rate = 3; %% These are for an exponential probability of success
min_exp_rate = 1; %% These are for an exponential probability of success
total_exp_step = 10;
exp_step = linspace(min_exp_rate,max_exp_rate,total_exp_step);

total_step = total_exp_step;

base_step_size = 10^-3;
base = 1.0000;

contract_tolerance = 10^-3;

for n=n_start:total_agents+n_start-1

    tolerance = 10^-15; % A tolerance which is used in the equilibrium calculation

    avg_hcprod_comparison = 0;
    avg_hcb_comparison = 0;
    avg_hcb_comparison_fm = 0;

    %G = zeros(n,n);
    %G(1:floor(n/2),1:floor(n/2)) = 5;
    %G(1:floor(n/2),floor(n/2)+1:n) = sqrt(5)/10;
    %G(floor(n/2)+1:n,1:floor(n/2)) = sqrt(5)/10;
    %G(1:n+1:end) = 0.01;

    G = zeros(n,n);
    G(1:floor(n/2),1:floor(n/2)) = 0.15*0.15;
    G(1:floor(n/2),floor(n/2)+1:n) = 0.3*0.15;
    G(floor(n/2)+1:n,1:floor(n/2)) = 0.3*0.15;
    G(1:n+1:end) = 0.3*0.3;

    %G = [[0,5,0.5];[5,0,0.5];[0,0.5,0.5]]; % 2 agents with a link and an isolated agent
    b = zeros(n,1);
    b(1:floor(n/2)) = 0.3;
    b(floor(n/2)+1:n) = 0.15;
    %b = [0.5;0.5;1]; % different standalone productivities

    avg_b_comparison_fm_sigstep = zeros(total_step,1);

    for prod_iter=1:total_step

        %step = sigmoid_step(prod_iter); %% Uncommment when using a sigmoid
        %step function
        step = exp_step(prod_iter); 
        fun_complementary = @(contract) -success_probability(team_performance(eqlb_action(contract,rand(n,1),beta,b,G,step),beta,b,G),step)*(1-sum(contract));
    
        for iter=1:total_simulations
    
            %% Hill climb with productivity and centrality        
            %% The algorithm runs as follows: Iterate over fraction of shares the principal would want to allocate.
            %% For each fixed share, initialize agents with a random contract. Consider a pair of agents with the highest
            %% and lowest productivity x centrality x marginal utility. Reallocate payments between these according to a step size.
            %% Iterate till you equalize the above object for every player.
            
            max_centrality_hillclimbprofits = 0;
            max_centrality_hillclimbprofits_fm = 0;
            max_b_hillclimbprofits = 0;
    
            for share_iter=1:share_discretization
                share_iter;
                share = share_list(share_iter); %% The fraction of the company agents receive from the principal
    
                rand_vec = zeros(1,n+1);
                rand_vec(2:n) = share*rand(1,n-1);
                rand_vec(n+1) = share;
                sorted_vec = sort(rand_vec);
                contract_init = sorted_vec(2:n+1) - sorted_vec(1:n); % An initialization to start the optimization problem. One observation was that for too small an initialization, the numerical optimization returned the initialization as is, i.e it was very close to a local minima
                contract_init = contract_init' ;
        
                %% Solve for an optimal contract using fmincon. Forget the hill climb for now
                constraint_matrix_eq = ones(1,n);
                constraint_const_vector(1) = share; % For the budget constraint. The rest are left to be zero because of the non-negativity constraints
                %constraint_ineq_vector(1) = share;
         
                x_mincon = fmincon(fun_complementary,contract_init,[],[],constraint_matrix_eq,constraint_const_vector,zeros(n,1),[],[],options);
        
                %% Find the optimal contract overall
        
                %x_mincon = fmincon(fun_complementary,contract_init,constraint_matrix_eq,constraint_ineq_vector,[],[],zeros(1,n),[],[],options); 
                contract_fm = x_mincon;
        
                equilibrium_action_fm = eqlb_action(contract_fm,rand(n,1),beta,b,G,step);
        
                y_fm = team_performance(equilibrium_action_fm,beta,b,G); % Team performance
                sp_fm = success_probability(y_fm,step); %P(Y)
                deriv_sp_fm = derivative_success_probability(y_fm,step);
        
                opt_profits_fm = sp_fm*(1-sum(contract_fm));
                if opt_profits_fm > max_centrality_hillclimbprofits_fm
                    max_centrality_hillclimbprofits_fm = opt_profits_fm;
                end
    
                %% Solve for a contract satisfying which equalizes b x marginal productivity space
    
                payment_team1 = share / (floor(n/2) + power(b(n)/b(1),4)*(n-floor(n/2)));
                payment_team2 = power(b(n)/b(1),4) * payment_team1;
    
                contract_b = contract_init;
                contract_b(1:floor(n/2)) = payment_team1;
                contract_b(floor(n/2)+1:n) = payment_team2;
                
                equilibrium_action_b = eqlb_action(contract_b,rand(n,1),beta,b,G,step);
                prev_equilibrium_action_b = equilibrium_action_b;
    
                y_b = team_performance(equilibrium_action_b,beta,b,G); % Team performance
                sp_b = success_probability(y_b,step); %P(Y)
                deriv_sp_b = derivative_success_probability(y_b,step);
    
                opt_profits_b = (1-share)*sp_b;
                if opt_profits_b > max_b_hillclimbprofits
                    max_b_hillclimbprofits = opt_profits_b;
                end
            end
            %avg_hcprod_comparison = avg_hcprod_comparison + max_centrality_hillclimbprofits/(max_productivity_hillclimbprofits*total_simulations);
            avg_hcb_comparison = avg_hcb_comparison + max_centrality_hillclimbprofits/(max_b_hillclimbprofits*total_simulations);
            avg_hcb_comparison_fm = avg_hcb_comparison_fm + max_centrality_hillclimbprofits_fm/(max_b_hillclimbprofits*total_simulations);
        end
        avg_b_comparison_fm_sigstep(prod_iter) = avg_hcb_comparison_fm;
    end
    %avg_b_comparison(agent_count) = avg_hcb_comparison;
    %avg_b_comparison_fm(agent_count) = avg_hcb_comparison_fm;
    agent_count = agent_count + 1;
end

figure(1)
%plot(n_start:n_start+total_agents-1,avg_b_comparison_fm,'linestyle','none',marker='o',markerfacecolor='blue');
%plot(sigmoid_step,avg_b_comparison_fm_sigstep,'linestyle','none',marker='o',markerfacecolor='blue');
plot(exp_step,avg_b_comparison_fm_sigstep,'linestyle','none',marker='o',markerfacecolor='blue');
hold on;
%plot(linspace(n_start,n_start+total_agents-1,1000),ones(1,1000),'LineStyle','--','LineWidth',2.0)
%plot(linspace(min_sigmoid_step,max_sigmoid_step,1000),ones(1,1000),'LineStyle','--','LineWidth',2.0);
plot(linspace(min_exp_rate,max_exp_rate,1000),ones(1,1000),'LineStyle','--','LineWidth',2.0);
%hold on;
%plot(n_start:n_start+total_agents-1,avg_b_comparison,'linestyle','none',marker='o',markerfacecolor='red');
hold off;
ax = gca;
xlabel("$\kappa$",'Interpreter','Latex','FontSize',15);
ylabel("Ratio of Revenue");
%legend({'Optimizer comparison','hill climb comparison'},'Interpreter','Latex','FontSize',15)
%legend({'Productivity Proportional Contract','Degree Proportional Contract'},'Interpreter','Latex','FontSize',15)
ax.XAxis.TickLength = [0 0.1]; 
ax.YAxis.TickLength = [0 0.1];
exportgraphics(gcf,'ratio_of_hillclimb.pdf','ContentType','vector');

function Y=team_performance(equilibrium_action,beta,b,G)
    Y = b'*equilibrium_action + (beta/2)*equilibrium_action'*G*equilibrium_action;
end

function G = weighted_graph(n,alpha,x_min)
    % Generate a wweighted graph where we use edge weights sampled from a
    % power law with parameters x_min and alpha
    
    edge_weights = sample_powerlaw(alpha,x_min,n*(n-1)/2);

    % Generate an adjacency matrix using these weights
    G = zeros(n,n);
    count = 1;
    for i=1:n
        for j=i+1:n
            G(i,j) = edge_weights(count);
            count = count + 1;
        end
    end
    G = G + G';
end

function G = random_simple_graph(deg)
    % Ensure even sum
    if mod(sum(deg), 2) ~= 0
        error('Sum of degree sequence must be even.');
    end
    
    n = length(deg);
    max_attempts = 20000;
    attempt = 0;
    
    while attempt < max_attempts
        attempt = attempt + 1;

        % Create stubs
        stubs = [];
        for i = 1:n
            stubs = [stubs; repmat(i, deg(i), 1)];
        end
        
        % Randomly permute stubs
        stubs = stubs(randperm(length(stubs)));

        % Pair stubs
        edges = [stubs(1:2:end), stubs(2:2:end)];

        % Check for self-loops or multi-edges
        if any(edges(:,1) == edges(:,2))
            continue;  % self-loop found
        end

        % Sort edges so (i,j) and (j,i) are the same
        sorted_edges = sort(edges, 2);
        [~, unique_idx] = unique(sorted_edges, 'rows');
        
        if length(unique_idx) ~= size(edges, 1)
            continue;  % multiple edge found
        end

        % Build and return graph
        G = graph(edges(:,1), edges(:,2));
        return;
    end

    error('Failed to generate a simple graph in %d attempts.', max_attempts);
end

function samples = sample_powerlaw(alpha, xmin, n)
    % sample_powerlaw samples from a continuous power-law distribution
    %   alpha: exponent (must be > 1)
    %   xmin: minimum value (support starts here)
    %   n: number of samples
    
    if min(alpha) <= 1
        error('Alpha must be greater than 1.');
    end
    
    % Generate uniform random numbers
    u = rand(n, 1);
    
    % Inverse transform sampling
    samples = xmin * (1 - u).^(-1 ./ (alpha - 1));
end

function samples = sample_discrete_power_law_bounded(alpha,d_min,d_max,n)
    % Sample n values from a power-law distribution with exponent alpha and minimum xmin
    k=d_min:d_max;
    p = k.^(-alpha);
    p = p / sum(p);
    samples = randsample(k,n,true,p);
end

function equilibrium = eqlb_action(contract,eqlb_initialization,beta,b,G,sig_step)
    n = length(contract);

    %% Our game is a potential game. Take best responses one agent at a time.
    %% How do you choose such an agent? Choose an agent randomly to best respond?
    %% Stopping criterion: norm of the first-order conditions is small
    
    count = 0;

    %% Solve for equilibrium by maximizing the potential function
    %% The contract design problem is a weighted potential game where the potential function
    %% phi = P(Y) - \sum_{i}a_{i}^2/\tau_{i}, and the weights are \tau_{i}

    a_init = eqlb_initialization;
    potential = @(x) -success_probability(team_performance(x,beta,b,G),sig_step) + sum(x.^2 ./ (2*utilities(contract)));

    equilibrium = fmincon(potential,a_init,[],[],[],[],zeros(n,1));
end

function eqlb_foc = eqlb_foc_norm(contract,action,beta,b,G)
    y = team_performance(action,beta,b,G);
    eqlb_foc = norm(derivative_success_probability(y)*(utilities(contract).*(b+beta*G*action))-action);
end

function sprob = success_probability(y,step)
    sprob = 1-exp(-step*y);
    %if y >= 1
    %    sprob = exp(-1)*(1-exp(-y+1))+1-exp(-1);
    %end
    %if y < 1
    %    sprob = 1-exp(-y);
    %end
    %sprob = 1/(1+exp(-(y-step)));
end

function dprob = derivative_success_probability(y,step)
    dprob = step*exp(-step*y);
    %if y >= 1
    %    dprob = exp(-1)*exp(-y+1);
    %end
    %if y < 1
    %    dprob = exp(-y);
    %end
    %dprob = exp(-(y-step))/power(1+exp(-(y-step)),2);
end

function util = utilities(contract)
    global e_factors;
    %util = 1-exp(-e_factors.*contract);
    %util = contract;
    util = sqrt(contract);
end

function m_util = marginal_utility(contract)
    global e_factors;
    %m_util = e_factors.*exp(-e_factors.*contract);
    m_util = 0.5 ./sqrt(contract);
end

function [c,ceq] = b_hillclimb(contract)
    c = [];
    n = length(contract);

    b = zeros(n,1);
    b(1:floor(n/2)) = 0.3;
    b(floor(n/2)+1:n) = 0.15;

    m_util = marginal_utility(contract);

    for iter=1:n-1
        ceq(iter) = b(iter)*m_util(iter) - b(iter+1)*m_util(iter+1); 
    end
end