%% VGG-16 Feature Extraction with Incremental PCA 
% 用途: 從 VGG-16 提取 nature images 特徵，並使用 incremental PCA 降維
% 流程:
%   1. 對每個 pooling 層 (pool1~pool5):
%      a. 訓練階段: 用所有受試者的圖片建立共用的 PCA 模型
%      b. 投影階段: 用訓練好的模型將特徵降維到 256 維
%   2. 分別儲存每個受試者在每個 pooling 層的降維特徵
% 輸入: Algonauts dataset 的訓練圖片
% 輸出: feature_map_dimreduced_subj{i}_pool{j}.mat (8 受試者 x 5 pooling 層 = 40 個檔案)
% 作者: I-HANG CHEN
% 日期: 2024 JUNE 21st

%% ========== 全域設定 ==========
% 資料路徑
DATA_FOLDER = './fMRI_Datasets/algonauts/data'; % actual path required
OUTPUT_FOLDER = '../data/feature_map';  % 輸出資料夾

% 受試者與 PCA 參數
NUM_SUBJECTS = 8;           % 受試者數量
NUM_POOLING_LAYERS = 5;     % VGG-16 的 pooling 層數量
BATCH_SIZE = 256;           % PCA 的 batch size
N_COMPONENTS = 256;         % 降維後的維度

% 確保輸出資料夾存在
if ~exist(OUTPUT_FOLDER, 'dir')
    mkdir(OUTPUT_FOLDER);
end

fprintf('========== VGG-16 特徵提取與降維 ==========\n');
fprintf('受試者數: %d\n', NUM_SUBJECTS);
fprintf('Pooling 層數: %d\n', NUM_POOLING_LAYERS);
fprintf('PCA 降維: %d 維\n', N_COMPONENTS);
fprintf('Batch size: %d\n\n', BATCH_SIZE);

%% ========== 載入 VGG-16 ==========
fprintf('載入 VGG-16 網路...\n');
net = vgg16;
input_size = net.Layers(1).InputSize;
fprintf('VGG-16 載入完成!\n\n');

%% ========== 第一步: 統計每個受試者的圖片數量 ==========
fprintf('========== 統計圖片數量 ==========\n');

num_images_per_subject = zeros(1, NUM_SUBJECTS);
img_files_all = cell(1, NUM_SUBJECTS);

for subj_id = 1:NUM_SUBJECTS
    img_folder = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), 'images');
    img_files = dir(fullfile(img_folder, '*.png'));
    num_images = length(img_files);
    
    num_images_per_subject(subj_id) = num_images;
    img_files_all{subj_id} = img_files;
    
    fprintf('受試者 %d: %d 張圖片\n', subj_id, num_images);
end

total_num_images = sum(num_images_per_subject);
fprintf('\n總圖片數: %d\n\n', total_num_images);

%% ========== 主迴圈: 處理每個 Pooling 層 ==========
for pool_id = 1:NUM_POOLING_LAYERS
    fprintf('========================================\n');
    fprintf('處理 pool%d 層\n', pool_id);
    fprintf('========================================\n');
    
    layer_name = sprintf('pool%d', pool_id);
    
    %% === 階段 1: 訓練 Incremental PCA 模型 ===
    fprintf('\n階段 1/2: 訓練 PCA 模型...\n');
    
    % 初始化 PCA 參數
    U = [];           % principal component
    S = [];           % singular value
    mu = [];          % mean
    n_samples_seen = 0;  % 已處理的樣本數
    
    % 初始化 batch buffer
    batch_buffer = [];
    batch_count = 0;
    
    % 所有受試者的圖片
    total_processed = 0;
    
    for subj_id = 1:NUM_SUBJECTS
        img_folder = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), 'images');
        img_files = img_files_all{subj_id};
        num_images = num_images_per_subject(subj_id);
        
        for img_id = 1:num_images
            % 載入並處理圖片
            filename = fullfile(img_folder, img_files(img_id).name);
            current_img = imread(filename);
            current_img = imresize(current_img, input_size(1:2), 'nearest');
            
            % 提取特徵
            feature = activations(net, current_img, layer_name);
            
            % flatten 特徵為 1D 向量
            feature_flat = reshape(feature, [1, numel(feature)]);
            
            % 加入 batch buffer
            batch_buffer = [batch_buffer; feature_flat];
            
            % 每當 batch 一滿就更新 PCA 模型
            if size(batch_buffer, 1) == BATCH_SIZE
                if n_samples_seen == 0
                    % 第一個 batch: 初始化 PCA
                    batch_trans = batch_buffer';  % [features x samples]
                    mu = mean(batch_trans, 2);
                    [U, S, ~] = svd(batch_trans - mu, 'econ');
                    n_samples_seen = BATCH_SIZE;
                else
                    % 後續 batch: 更新
                    [U, S, mu, n_samples_seen] = incrementalPCA(...
                        batch_buffer', U, S, mu, n_samples_seen, N_COMPONENTS);
                end
                
                % 清空buffer
                batch_buffer = [];
                batch_count = batch_count + 1;
                
                % 顯示進度
                total_processed = total_processed + BATCH_SIZE;
                if mod(batch_count, 10) == 0
                    fprintf('  已處理 %d/%d 張圖片 (%.1f%%)\n', ...
                        total_processed, total_num_images, ...
                        100 * total_processed / total_num_images);
                end
            end
        end
    end
    
    % 處理最後不滿一個 batch 的資料
    if ~isempty(batch_buffer)
        if n_samples_seen == 0
            % 如果總圖片數少於一個 batch
            batch_trans = batch_buffer';
            mu = mean(batch_trans, 2);
            [U, S, ~] = svd(batch_trans - mu, 'econ');
            n_samples_seen = size(batch_buffer, 1);
        else
            [U, S, mu, n_samples_seen] = incrementalPCA(...
                batch_buffer', U, S, mu, n_samples_seen, N_COMPONENTS);
        end
        total_processed = total_processed + size(batch_buffer, 1);
    end
    
    fprintf('  PCA 訓練完成! 共處理 %d 張圖片\n', n_samples_seen);
    
    % 確保只保留前 N_COMPONENTS 個主成分
    U = U(:, 1:N_COMPONENTS);
    S = S(1:N_COMPONENTS, 1:N_COMPONENTS);
    
    %% === 階段 2: 使用 PCA 模型投影所有圖片 ===
    fprintf('\n階段 2/2: 投影特徵到低維空間...\n');
    
    % 對每個受試者分別處理並儲存
    for subj_id = 1:NUM_SUBJECTS
        fprintf('  處理受試者 %d/%d...', subj_id, NUM_SUBJECTS);
        
        img_folder = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), 'images');
        img_files = img_files_all{subj_id};
        num_images = num_images_per_subject(subj_id);
        
        % Allocate the space
        reduced_features = zeros(num_images, N_COMPONENTS);
        
        % 處理該受試者的所有圖片
        for img_id = 1:num_images
            % 載入並處理圖片
            filename = fullfile(img_folder, img_files(img_id).name);
            current_img = imread(filename);
            current_img = imresize(current_img, input_size(1:2), 'nearest');
            
            % 提取特徵
            feature = activations(net, current_img, layer_name);
            
            % flatten特徵
            feature_flat = reshape(feature, [1, numel(feature)])';
            
            % PCA 投影: y = U' * (x - mu)
            reduced_feature = U' * (feature_flat - mu);
            
            % 儲存降維後的特徵
            reduced_features(img_id, :) = reduced_feature';
        end
        
        % 儲存該受試者的降維特徵
        output_filename = fullfile(OUTPUT_FOLDER, ...
            sprintf('feature_map_dimreduced_subj%d_pool%d.mat', subj_id, pool_id));
        
        % 為了與 compute_correlations.m 相容，使用動態變數名
        feature_map_data = reduced_features;
        save(output_filename, 'feature_map_data');
        
        fprintf(' 完成! (%d x %d)\n', size(reduced_features, 1), size(reduced_features, 2));
    end
    
    % 儲存 PCA 模型參數
    pca_model_filename = fullfile(OUTPUT_FOLDER, sprintf('pca_model_pool%d.mat', pool_id));
    save(pca_model_filename, 'U', 'S', 'mu', 'n_samples_seen');
    
    fprintf('\npool%d 層處理完成!\n\n', pool_id);
end

fprintf('========================================\n');
fprintf('所有 pooling 層處理完成!\n');
fprintf('========================================\n');
fprintf('輸出檔案位置: %s\n', OUTPUT_FOLDER);
fprintf('總共產生 %d 個特徵檔案\n', NUM_SUBJECTS * NUM_POOLING_LAYERS);
fprintf('總共產生 %d 個 PCA 模型檔案\n\n', NUM_POOLING_LAYERS);