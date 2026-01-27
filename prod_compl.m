n = 4; % Number of agents
n_diff = 2; % Number of agents that receive heterogenous payments
%n=3; % Number of guests

tolerance = 10^-15; % A tolerance which is used in the equilibrium calculation
%options = optimset('Display','off');
options = optimoptions('fmincon', 'Display', 'off','FiniteDifferenceType','central');
%options = optimoptions('fmincon','Display','off','Algorithm','sqp');
%rho = 1;

total_iter = 100;
%k_list = linspace(0.9,1,total_iter);
%beta_list = linspace(0.9,1,total_iter);

%k = 0.9*ones(n,1);
%beta = ones(n,1);
%beta = [0.5;0.8;1];
%k = [1;0.8;0.5];
k = zeros(n,1);
k(1:floor(n/2)) = 0.5;
k(floor(n/2)+1:n) = 0.25;

beta = zeros(n,1);
beta(1:floor(n/2)) = 0.25;
beta(floor(n/2)+1:n) = 0.5;

opt_solns_rhovary = zeros(n,total_iter);
rho_list = linspace(0,10,total_iter);

specrad_list = zeros(1,total_iter);
 
total_simulations = 20;

ratio_payment = zeros(n_diff*(n_diff-1)/2,total_iter);
%ratio_payment = zeros(1,total_iter);

heuristic_grid_size = 10;
share_grid = linspace(0,1,heuristic_grid_size);

sig_step = 3;

tot_share_sims = 1;
heuristic_profit_comparison = zeros(1,total_iter);

tp_rhovary = zeros(1,total_iter);

for iter=1:total_iter
    rho = rho_list(iter)
    fun_complementary = @(x) -(1-(n/2)*sum(x))*success_probability(team_performance(eqlb_action_potential(x,rand(n,1),rho,k,beta,sig_step),rho,k,beta),sig_step);
    
    opt_contract_avg_diff = 0;
    opt_contract_avg = 0;
    avg_spillover = 0;
    opt_profits = 0;
    avg_tp = 0;

    for sim=1:total_simulations   
        sim
        scale = 0.5*rand(); % Can choose any scaling in [0,0.5].
        rand_vec = zeros(1,n_diff+1);
        rand_vec(2:n_diff) = scale*rand(1,n_diff-1);
        rand_vec(n_diff+1) = scale;
        sorted_vec = sort(rand_vec);
        x_init = sorted_vec(2:n_diff+1) - sorted_vec(1:n_diff); % An initialization to start the optimization problem. One observation was that for too small an initialization, the numerical optimization returned the initialization as is, i.e it was very close to a local minima
        x_init = x_init' ;
        x_mincon = fmincon(fun_complementary,x_init,[],[],[],[],zeros(n_diff,1),[],[],options);
        opt_contract_avg_diff = opt_contract_avg_diff + x_mincon;
    
        for i=1:n_diff-1
            for j=i+1:n_diff
                ratio_payment(n_diff*(n_diff-1)/2 - (n_diff-i)*(n_diff-i+1)/2 + j - i,iter) = ratio_payment(n_diff*(n_diff-1)/2 - (n_diff-i)*(n_diff-i+1)/2 + j - i,iter) + x_mincon(i)/x_mincon(j);
            end
        end
        profits = -fun_complementary(x_mincon);
        opt_profits = opt_profits + profits;
    
        %% Compute the spectral radius of the spillover matrix at the optimal contract
        actions = eqlb_action_potential(x_mincon,rand(n,1),rho,k,beta,sig_step);
        tp = team_performance(actions,rho,k,beta);
        avg_tp = avg_tp + tp;

        contract = zeros(n,1);
        contract(1:floor(n/2)) = x_mincon(1);
        contract(floor(n/2)+1:n) = x_mincon(2);

        opt_contract_avg = opt_contract_avg + contract;

        spillover_mat = derivative_success_probability(tp,sig_step)*rho*diag(utilities(contract))*(beta*beta'); 
        avg_spillover = avg_spillover + norm(spillover_mat);
    end
    %payoff_agt_kvary(:,iter) = utilities(x_mincon) * success_probability(opt_team_performance,1) - actions.^2/2;
    opt_solns_rhovary(:,iter) = opt_contract_avg / total_simulations;
    specrad_list(iter) = avg_spillover/total_simulations;
    opt_profits = opt_profits / total_simulations;
    tp_rhovary(iter) = avg_tp / total_simulations;

    %ratio_payment = ratio_payment ./ total_simulations;

%     avg_heuristic_profits = 0;
%     for share_sims=1:tot_share_sims
%         opt_heuristic_profits = 0;
%         for share_iter=1:heuristic_grid_size
%             share = share_grid(share_iter);
%             contract_heur_diff = zeros(n_diff,1);
%             denom = k(1)^4 + k(n)^4;
%             contract_heur_diff(1) = share*(k(1)^4/denom)*(2/n);
%             contract_heur_diff(2) = share*(k(n)^4/denom)*(2/n);
%             heuristic_profits = -fun_complementary(contract_heur_diff);
%             if heuristic_profits > opt_heuristic_profits
%                 opt_heuristic_profits = heuristic_profits;
%             end
%         end
%         avg_heuristic_profits = avg_heuristic_profits + opt_heuristic_profits/tot_share_sims;
%     end
%     heuristic_profit_comparison(iter) = opt_profits/avg_heuristic_profits;
end
ratio_payment = ratio_payment ./ total_simulations;

figure(1);
plot(rho_list(1:total_iter/2),ratio_payment(1,1:total_iter/2),':','LineWidth',2);
%hold on
%plot(rho_list(1:total_iter/2),ratio_payment(2,1:total_iter/2),':','LineWidth',2);
%hold on
%plot(rho_list(1:total_iter/2),ratio_payment(3,1:total_iter/2),'-.','LineWidth',2);
%hold off
%legend({'$\tau_{1} / \tau_{2}$','$\tau_{1} / \tau_{3}$','$\tau_{2} / \tau_{3}$'},'Interpreter','latex')
xlabel("$\rho$",'Interpreter','latex')
ylabel("Ratio of payments")

figure(2);
plot(rho_list(total_iter/2+1:total_iter),ratio_payment(1,total_iter/2+1:total_iter),':','LineWidth',2);
%hold on
%plot(rho_list(total_iter/2+1:total_iter),ratio_payment(2,total_iter/2+1:total_iter),':','LineWidth',2);
%hold on
%plot(rho_list(total_iter/2+1:total_iter),ratio_payment(3,total_iter/2+1:total_iter),'-.','LineWidth',2);
%hold off
%legend({'$\tau_{1} / \tau_{2}$','$\tau_{1} / \tau_{3}$','$\tau_{2} / \tau_{3}$'},'Interpreter','latex');
xlabel("$\rho$",'Interpreter','latex');
ylabel("Ratio of payments");

figure(3);
plot(rho_list,specrad_list,'LineWidth',2);
xlabel("$\rho$",'Interpreter','latex');
ylabel("Spectral radius");

% figure(4);
% plot(rho_list,heuristic_profit_comparison,'LineWidth',2);
% xlabel("$\rho$",'Interpreter','latex');
% ylabel("Ratio of Revenue");

function Y=team_performance(equilibrium_action,rho,k,beta)
    %Y = b'*equilibrium_action + (beta/2)*equilibrium_action'*G*equilibrium_action;
    Y = k'*equilibrium_action + (rho/2)*power(beta'*equilibrium_action,2);
end

function equilibrium = eqlb_action_potential(contract_diff,eqlb_initialization,rho,k,beta,sig_step)
    n = length(k);
    contract = zeros(n,1);
    contract(1:floor(n/2)) = contract_diff(1);
    contract(floor(n/2)+1:n) = contract_diff(2);

    %options = optimoptions('fmincon', 'Display', 'off');
    options = optimoptions('fmincon','Display','off','FiniteDifferenceType','central');

    %% Our game is a potential game. Take best responses one agent at a time.
    %% How do you choose such an agent? Choose an agent randomly to best respond?
    %% Stopping criterion: norm of the first-order conditions is small
    
%     count = 0;
% 
%     %% Solve for equilibrium by maximizing the potential function
%     %% The contract design problem is a weighted potential game where the potential function
%     %% phi = P(Y) - \sum_{i}a_{i}^2/\tau_{i}, and the weights are \tau_{i}
% 
%     %a_init = eqlb_initialization;
%     %potential = @(x) -success_probability(team_performance(x,rho,k,beta),sig_step) + sum(x.^2 ./ (2*utilities(contract)));
% 
%     %equilibrium = fmincon(potential,a_init,[],[],[],[],zeros(n,1));
% 
%     %% Compute equilibrium using a fixed point subroutine
%     %% Compute limits within which you will do binary search
%     y_max = 4;
%     y_min = 0;
% 
%     spillover_max = derivative_success_probability(y_max,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%     eqlb_max = derivative_success_probability(y_max,sig_step)*((eye(n)-spillover_max)^-1)*(utilities(contract).*k);
%     a_ymax = team_performance(eqlb_max,rho,k,beta);
%     while a_ymax > y_max
%         y_max = 2*y_max;
%         spillover_max = derivative_success_probability(y_max,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%         eqlb_max = derivative_success_probability(y_max,sig_step)*((eye(n)-spillover_max)^-1)*(utilities(contract).*k);
%         a_ymax = team_performance(eqlb_max,rho,k,beta);
%     end
% 
%     y_max_temp = y_max;
% 
%     spillover = derivative_success_probability(y_min,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%     while norm(spillover) > 1
%         y_min = y_min + (y_max_temp-y_min)/2;
%         spillover = derivative_success_probability(y_min,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%         count = count + 1;
%         if count > 1000
%             %disp("getting stuck here")
%             break
%         end
%     end
% 
%     eqlb = derivative_success_probability(y_min,sig_step)*((eye(n)-spillover)^-1)*(utilities(contract).*k);
%     a_y = team_performance(eqlb,rho,k,beta);
%     count1 = 0;
%     count2 = 0;
%     while y_min > a_y
%         y_max_temp = y_min;
%         y_min  = 0;
%         spillover = derivative_success_probability(y_min,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%         count2 = 0;
%         while norm(spillover) > 1
%             y_min = y_min + (y_max_temp-y_min)/2;
%             spillover = derivative_success_probability(y_min,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%             count2 = count2 + 1;
%             if count2 > 1000
%                 %disp("getting stuck in the first loop");
%                 %norm(spillover)
%                 break
%             end
%         end    
%         eqlb = derivative_success_probability(y_min,sig_step)*((eye(n)-spillover)^-1)*(utilities(contract).*k);
%         a_y = team_performance(eqlb,rho,k,beta);
%         count1 = count1 + 1;
%         if count1 > 1000
%             %disp("getting stuck in the second loop");
%             break
%         end
%     end
% 
%     if y_min - a_y > 0
%         disp("Hello")
%     end
% 
%     %% Now perform a binary search between y_min and y_max
%     count3 = 0;
%     while (y_max-y_min)^2 > 10^-6
%         y_temp = (y_min+y_max)/2;
%         spillover = derivative_success_probability(y_temp,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%         eqlb = derivative_success_probability(y_temp,sig_step)*((eye(n)-spillover)^-1)*(utilities(contract).*k);
%         a_y = team_performance(eqlb,rho,k,beta);
%         if a_y > y_temp
%             y_min = y_temp;
%         else
%             y_max = y_temp;
%         end
%         %(y_max-y_min)^2
%         count3 = count3 + 1;
%         if count3 > 10000
%             %disp("getting stuck in the final loop");
%             break
%         end
%     end
% 
%     y_temp = (y_min+y_max)/2;
%     spillover = derivative_success_probability(y_temp,sig_step)*rho*diag(utilities(contract))*(beta*beta');
%     equilibrium = derivative_success_probability(y_temp,sig_step)*((eye(n)-spillover)^-1)*(utilities(contract).*k);
% 
%     if count>1000 || count1>1000 || count2 > 1000 || count3 > 1000
%         % Solve for equilibrium using the potential method
%         disp("potential method")
%         a_init = eqlb_initialization;
%         potential = @(x) -success_probability(team_performance(x,rho,k,beta),sig_step) + sum(x.^2 ./ (2*utilities(contract)));
%         equilibrium = fmincon(potential,a_init,[],[],[],[],zeros(n,1));
%     end

    %a_init = eqlb_initialization;
    a_init = zeros(n,1);
    potential = @(x) -success_probability(team_performance(x,rho,k,beta),sig_step) + sum(x.^2 ./ (2*utilities(contract)));
    equilibrium = fmincon(potential,a_init,[],[],[],[],zeros(n,1),[],[],options);
end

function eqlb_foc = eqlb_foc_norm(contract,action,beta,b,G)
    y = team_performance(action,beta,b,G);
    eqlb_foc = norm(derivative_success_probability(y)*(utilities(contract).*(b+beta*G*action))-action);
end

function sprob = success_probability(y,step)
    %sprob = 1-exp(-y);
    %sprob = 0.5*y;
    %if y >= 1
    %    sprob = exp(-1)*(1-exp(-y+1))+1-exp(-1);
    %end
    %if y < 1
    %    sprob = 1-exp(-y);
    %end
    sprob = 1/(1+exp(-(y-step)));
end

function dprob = derivative_success_probability(y,step)
    %dprob = exp(-y);
    %dprob = 0.5;
    %if y >= 1
    %    dprob = exp(-1)*exp(-y+1);
    %end
    %if y < 1
    %    dprob = exp(-y);
    %end
    dprob = exp(-(y-step))/power(1+exp(-(y-step)),2);
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
    m_util = 0.5 ./ sqrt(contract);
    %n = length(contract);
    %m_util = ones(n,1);
end