function [U_new, S_new, mu_new, n_samples_seen] = incrementalPCA(X, U, S, mu, ...
    n_samples_seen, n_components, energy_thresh)
% INCREMENTALPCA - 增量式主成分分析 (Incremental PCA)
%
% 功能說明:
%   給定已分解的資料矩陣 A [d x n]，當新資料 X [d x m] 到達時，
%   此演算法可以有效計算合併矩陣 [A X] 的 PCA ，不須重新計算整個 SVD。
%   用於處理超越硬體能handle的大型資料集。
%
% 演算法來源:
%   D. Ross et al., "Incremental Learning for Robust Visual Tracking"
%   International Journal of Computer Vision, 2008
%   https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf
%
% 輸入參數:
%   X                - 新資料矩陣 [d x m]，d=特徵維度, m=新樣本數
%   U                - 先前計算的主成分 [d x k]
%   S                - 先前計算的奇異值對角矩陣 [k x k]
%   mu               - 先前資料的樣本平均 [d x 1]
%   n_samples_seen   - 已處理的總樣本數 (用於計算 U 和 S)
%   n_components     - (可選) 保留的主成分數量，預設為保留全部
%                      使用 [] 表示預設值
%   energy_thresh    - (可選, 0~1) 當 n_components==[] 時，自動選擇
%                      能保留此比例變異數的主成分數量
%
% 輸出參數:
%   U_new            - 更新後的主成分 [d x n_components]
%   S_new            - 更新後的奇異值對角矩陣 [n_components x n_components]
%   mu_new           - 更新後的樣本平均 [d x 1]
%   n_samples_seen   - 更新後的總樣本數
%
% 使用範例 1: 基本用法
%   % 初始化 PCA
%   d = 5;                          % 特徵維度
%   n = 20;                         % 初始樣本數
%   A = rand(d, n);                 % 初始資料
%   mu = mean(A, 2);                % 計算平均
%   [U, S, ~] = svd(A - mu, 'econ'); % 初始 SVD
%   
%   % 新資料到達
%   X = rand(d, 5);                 % 5 個新樣本
%   [U, S, mu, n] = incrementalPCA(X, U, S, mu, n, []);
%   
%   % 可以重複使用
%   X_new = rand(d, 10);            % 更多新資料
%   [U, S, mu, n] = incrementalPCA(X_new, U, S, mu, n, []);
%
% 使用範例 2: 限制主成分數量
%   % 只保留前 3 個主成分
%   n_components = 3;
%   [U, S, mu, n] = incrementalPCA(X, U, S, mu, n, n_components);
%
% 使用範例 3: 根據變異數比例選擇主成分
%   % 保留 95% 的變異數
%   energy_thresh = 0.95;
%   [U, S, mu, n] = incrementalPCA(X, U, S, mu, n, [], energy_thresh);
%
% 注意事項:
%   - 輸入資料 X 應為 [特徵 x 樣本] 格式
%   - 此函數可以連續呼叫，每次的輸出可作為下次的輸入
%   - 演算法的時間複雜度遠低於重新計算完整 SVD
%
% 參見:
%   svd, pca

% 作者: 改編自 D. Ross et al. (2008)
% 日期: 2024 JULY 12th

%% ========== 輸入驗證 ==========
if nargin < 6
    n_components = [];
end

if nargin < 7
    energy_thresh = [];
end

% 驗證維度一致性
[d_U, ~] = size(U);
[d_mu, ~] = size(mu);
[d_X, m] = size(X);

if d_U ~= d_mu || d_U ~= d_X
    error('incrementalPCA:DimensionMismatch', ...
        '輸入矩陣的特徵維度不一致: U(%d), mu(%d), X(%d)', d_U, d_mu, d_X);
end

%% ========== incremental PCA ==========

n = n_samples_seen;  % 已處理的樣本數

% 步驟 1: 計算新資料的平均值
mu_X = mean(X, 2);

% 步驟 2: 更新加權平均
mu_new = (n / (n + m)) * mu + (m / (n + m)) * mu_X;

% 步驟 3: 建構增廣矩陣
% 包含: 1) 新資料的去中心化版本
%       2) 平均值修正項 (用於保持總體平均正確)
mean_correction = sqrt(n * m / (n + m)) * (mu_X - mu);
X_augmented = [X - mu_X, mean_correction];

% 步驟 4: 正交化增廣矩陣
% 將 [U*S, X_augmented] 進行 QR 分解
[U_intermediate, R] = qr([U * S, X_augmented], 0);

% 步驟 5: 對 R 進行 SVD
% 這個 SVD 的維度遠小於原始資料，因此計算效率高
[U_tilde, S_new, ~] = svd(R, 'econ');

% 步驟 6: 更新主成分
U_new = U_intermediate * U_tilde;

%% ========== 降維處理 ==========

% 根據參數決定保留多少主成分
if isempty(n_components) && ~isempty(energy_thresh) && energy_thresh < 1
    % 根據能量閾值自動選擇主成分數量
    singular_values = diag(S_new);
    cumulative_energy = cumsum(singular_values) / sum(singular_values);
    n_components = find(cumulative_energy >= energy_thresh, 1);
    
    if isempty(n_components)
        % 如果沒找到，保留全部
        n_components = size(S_new, 1);
    end
elseif isempty(n_components)
    % 保留全部主成分
    n_components = size(S_new, 1);
end

% 限制主成分數量不超過可用數量
n_components = min(n_components, size(S_new, 1));

% 只保留前 n_components 個主成分
U_new = U_new(:, 1:n_components);
S_new = S_new(1:n_components, 1:n_components);

%% ========== 更新樣本計數 ==========
n_samples_seen = n + m;

end