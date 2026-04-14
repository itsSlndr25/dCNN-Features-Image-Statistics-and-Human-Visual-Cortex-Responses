%% Correlation Analysis 
% 用於計算 VGG-16 各layer、fMRI ROI 與image statistics間的相關性
% 分析三種關係:
%   1. Layer X Stat: VGG-16各層 vs image statistics
%   2. ROI X Stat: fMRI各腦區 vs image statistics  
%   3. ROI X Layer: fMRI各腦區 vs VGG-16各層
% 作者: I-HANG CHEN
% 日期: 2024 JULY 12th

%% ========== 全域設定 ==========
% 受試者與資料結構參數
NUM_SUBJECTS = 8;           % 受試者數量
NUM_POOLING_LAYERS = 5;     % VGG-16 pooling層數
NUM_STATS = 4;              % image statistics數 (mean, contrast, skewness, slope)
NUM_ROIS = 7;               % fMRI ROI數 (V1v, V1d, V2v, V2d, V3v, V3d, hV4)
NUM_FEATURES = 256;         % 降維後的特徵數

% image statistics及腦區名稱
STAT_NAMES = {'mean', 'contrast', 'skewness', 'fft_slope'};
ROI_NAMES = {'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'};

% fMRI 資料路徑
MRI_DATA_FOLDER = './fMRI_Datasets/algonauts/data'; % actual path required

% 添加路徑
addpath('../data/feature_map');
addpath('../data/ROIs');
addpath('../readNPY');
addpath('../data/img_stat');

%% ========== Section 1: Layer X Stat (Log-transformed) ==========
fprintf('========== 開始分析 Layer X Stat (Log) ==========\n');

% Allocate space: pooling層 X image statistics X 受試者
max_correlation = zeros(NUM_POOLING_LAYERS, NUM_STATS, NUM_SUBJECTS);
mean_correlation = zeros(NUM_POOLING_LAYERS, NUM_STATS, NUM_SUBJECTS);

for k = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', k, NUM_SUBJECTS);
    
    % 載入該受試者的image statistics資料 (log-transformed)
    img_stats = loadImageStats(k, true); % true = log版本
    
    % 初始化計數器
    pool_idx = 1;      % pooling層索引
    stat_idx = 1;      % 統計特徵索引
    
    % run through所有 pooling層 X 統計特徵 的組合 (5 x 4 = 20)
    for h = 1:NUM_POOLING_LAYERS * NUM_STATS
        
        % 選擇當前的統計特徵
        current_stat = img_stats{stat_idx};
        
        % 載入對應的降維後的 VGG-16 特徵圖
        feature_map = loadFeatureMap(k, pool_idx);
        
        % 計算該image statistics與所有特徵維度的相關性
        corr_matrix = zeros(1, NUM_FEATURES);
        for i = 1:NUM_FEATURES
            corr_coef = corrcoef(current_stat, feature_map(:, i));
            corr_matrix(i) = corr_coef(1, 2);
        end
        
        % 記錄最大相關性(max)和平均相關性(mean) (絕對值)
        max_correlation(pool_idx, stat_idx, k) = max(abs(corr_matrix));
        mean_correlation(pool_idx, stat_idx, k) = mean(abs(corr_matrix));
        
        % 更新計數器 (每5個pooling層後換下一個image statistics)
        pool_idx = pool_idx + 1;
        if pool_idx > NUM_POOLING_LAYERS
            pool_idx = 1;
            stat_idx = stat_idx + 1;
        end
    end
end

% 儲存結果
save('correlation_LayerxStat_max.mat', 'max_correlation');
save('correlation_LayerxStat_mean.mat', 'mean_correlation');
fprintf('Layer X Stat (Log) 完成!\n\n');

% 清除變數但保留全域設定
clearvars -except NUM_* STAT_NAMES ROI_NAMES MRI_DATA_FOLDER;

%% ========== Section 2: Layer X Stat (Raw) ==========
fprintf('========== 開始分析 Layer X Stat (Raw) ==========\n');

% Allocate space
max_correlation = zeros(NUM_POOLING_LAYERS, NUM_STATS, NUM_SUBJECTS);
mean_correlation = zeros(NUM_POOLING_LAYERS, NUM_STATS, NUM_SUBJECTS);

for k = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', k, NUM_SUBJECTS);
    
    % 載入該受試者的image statistics data (raw版)
    img_stats = loadImageStats(k, false); % false = raw版
    
    % 初始化計數器
    pool_idx = 1;
    stat_idx = 1;
    
    % run through所有組合
    for h = 1:NUM_POOLING_LAYERS * NUM_STATS
        
        % 選擇當前的image statistics
        current_stat = img_stats{stat_idx};
        
        % 載入對應的 VGG-16 特徵圖
        feature_map = loadFeatureMap(k, pool_idx);
        
        % 計算相關性
        corr_matrix = zeros(1, NUM_FEATURES);
        for i = 1:NUM_FEATURES
            corr_coef = corrcoef(current_stat, feature_map(:, i));
            corr_matrix(i) = corr_coef(1, 2);
        end
        
        % 記錄結果
        max_correlation(pool_idx, stat_idx, k) = max(abs(corr_matrix));
        mean_correlation(pool_idx, stat_idx, k) = mean(abs(corr_matrix));
        
        % 更新計數器
        pool_idx = pool_idx + 1;
        if pool_idx > NUM_POOLING_LAYERS
            pool_idx = 1;
            stat_idx = stat_idx + 1;
        end
    end
end

% 儲存結果
save('correlation_rawLayerxStat_max.mat', 'max_correlation');
save('correlation_rawLayerxStat_mean.mat', 'mean_correlation');
fprintf('Layer X Stat (Raw) 完成!\n\n');

% 清除變數
clearvars -except NUM_* STAT_NAMES ROI_NAMES MRI_DATA_FOLDER;

%% ========== Section 3: ROI X Stat (Log-transformed) ==========
fprintf('========== 開始分析 ROI X Stat (Log) ==========\n');

% Allocate space: image statistics X ROI X 受試者
max_correlation = zeros(NUM_STATS, NUM_ROIS, NUM_SUBJECTS);
mean_correlation = zeros(NUM_STATS, NUM_ROIS, NUM_SUBJECTS);

for k = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', k, NUM_SUBJECTS);
    
    % 載入該受試者的 fMRI 資料和 ROI mask
    [fMRI_ROIs, ~] = loadfMRIData(k, MRI_DATA_FOLDER);
    
    % 載入該受試者的image statistics data (log版)
    img_stats = loadImageStats(k, true);
    
    % 初始化計數器
    roi_idx = 1;
    stat_idx = 1;
    
    % run through 所有 image statistics X ROI 的組合 (4 x 7 = 28)
    for h = 1:NUM_STATS * NUM_ROIS
        
        % 選擇當前的 ROI 資料
        current_roi_data = fMRI_ROIs{roi_idx};
        
        % 選擇當前的 image statistics
        current_stat = img_stats{stat_idx};
        
        % 計算該 image statistics 與所有 voxels 的相關性
        num_voxels = size(current_roi_data, 2);
        corr_matrix = zeros(1, num_voxels);
        
        for i = 1:num_voxels
            voxel_timeseries = current_roi_data(:, i); % 第 i 個 voxel 的時間序列
            corr_coef = corrcoef(current_stat, voxel_timeseries);
            corr_matrix(i) = corr_coef(1, 2);
        end
        
        % 記錄結果
        max_correlation(stat_idx, roi_idx, k) = max(abs(corr_matrix));
        mean_correlation(stat_idx, roi_idx, k) = mean(abs(corr_matrix));
        
        % 更新計數器 (每4個image statistics後換下一個ROI)
        stat_idx = stat_idx + 1;
        if stat_idx > NUM_STATS
            stat_idx = 1;
            roi_idx = roi_idx + 1;
        end
    end
end

% 儲存結果
save('correlation_ROIxStat_max.mat', 'max_correlation');
save('correlation_ROIxStat_mean.mat', 'mean_correlation');
fprintf('ROI X Stat (Log) 完成!\n\n');

% 清除變數
clearvars -except NUM_* STAT_NAMES ROI_NAMES MRI_DATA_FOLDER;

%% ========== Section 4: ROI X Stat (Raw) ==========
fprintf('========== 開始分析 ROI X Stat (Raw) ==========\n');

% Allocate space
max_correlation = zeros(NUM_STATS, NUM_ROIS, NUM_SUBJECTS);
mean_correlation = zeros(NUM_STATS, NUM_ROIS, NUM_SUBJECTS);

for k = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', k, NUM_SUBJECTS);
    
    % 載入該受試者的 fMRI 資料和 ROI mask
    [fMRI_ROIs, ~] = loadfMRIData(k, MRI_DATA_FOLDER);
    
    % 載入該受試者的image statistics資料 (raw版)
    img_stats = loadImageStats(k, false);
    
    % 初始化計數器
    roi_idx = 1;
    stat_idx = 1;
    
    % 遍歷所有組合
    for h = 1:NUM_STATS * NUM_ROIS
        
        % 選擇當前的 ROI 資料
        current_roi_data = fMRI_ROIs{roi_idx};
        
        % 選擇當前的image statistics
        current_stat = img_stats{stat_idx};
        
        % 計算相關性
        num_voxels = size(current_roi_data, 2);
        corr_matrix = zeros(1, num_voxels);
        
        for i = 1:num_voxels
            voxel_timeseries = current_roi_data(:, i);
            corr_coef = corrcoef(current_stat, voxel_timeseries);
            corr_matrix(i) = corr_coef(1, 2);
        end
        
        % 記錄結果
        max_correlation(stat_idx, roi_idx, k) = max(abs(corr_matrix));
        mean_correlation(stat_idx, roi_idx, k) = mean(abs(corr_matrix));
        
        % 更新計數器
        stat_idx = stat_idx + 1;
        if stat_idx > NUM_STATS
            stat_idx = 1;
            roi_idx = roi_idx + 1;
        end
    end
end

% 儲存結果
save('correlation_rawROIxStat_max.mat', 'max_correlation');
save('correlation_rawROIxStat_mean.mat', 'mean_correlation');
fprintf('ROI X Stat (Raw) 完成!\n\n');

% 清除變數
clearvars -except NUM_* STAT_NAMES ROI_NAMES MRI_DATA_FOLDER;

%% ========== Section 5: ROI X Layer ==========
fprintf('========== 開始分析 ROI X Layer ==========\n');

% 預分配空間: pooling層 X ROI X 受試者
max_correlation = zeros(NUM_POOLING_LAYERS, NUM_ROIS, NUM_SUBJECTS);
mean_correlation = zeros(NUM_POOLING_LAYERS, NUM_ROIS, NUM_SUBJECTS);

for k = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', k, NUM_SUBJECTS);
    
    % 載入該受試者的 fMRI 資料和 ROI mask
    [fMRI_ROIs, ~] = loadfMRIData(k, MRI_DATA_FOLDER);
    
    % 初始化計數器
    roi_idx = 1;
    pool_idx = 1;
    
    % run through 所有 pooling層 X ROI 的組合 (5 x 7 = 35)
    for h = 1:NUM_POOLING_LAYERS * NUM_ROIS
        
        % 選擇當前的 ROI 資料
        current_roi_data = fMRI_ROIs{roi_idx};
        num_voxels = size(current_roi_data, 2);
        
        % 載入對應的 VGG-16 特徵圖
        feature_map = loadFeatureMap(k, pool_idx);
        
        % 計算該 pooling 層與所有 voxels 的相關性
        % Note: 這裡是雙重迴圈，計算每個 voxel 與每個特徵維度的相關性
        corr_matrix = zeros(num_voxels, NUM_FEATURES);
        
        for i = 1:num_voxels
            voxel_timeseries = current_roi_data(:, i); % 第 i 個 voxel
            for j = 1:NUM_FEATURES
                feature_timeseries = feature_map(:, j); % 第 j 個特徵維度
                corr_coef = corrcoef(feature_timeseries, voxel_timeseries);
                corr_matrix(i, j) = corr_coef(1, 2);
            end
        end
        
        % 記錄結果
        max_correlation(pool_idx, roi_idx, k) = max(abs(corr_matrix), [], 'all');
        mean_correlation(pool_idx, roi_idx, k) = mean(abs(corr_matrix), 'all');
        
        % 儲存完整的相關性矩陣
        % filename = sprintf('correlation_subj0%d_pool%d_roi%d.mat', k, pool_idx, roi_idx);
        % save(['./corr/' filename], 'corr_matrix');
        
        % 更新計數器 (每5個pooling層後換下一個ROI)
        pool_idx = pool_idx + 1;
        if pool_idx > NUM_POOLING_LAYERS
            pool_idx = 1;
            roi_idx = roi_idx + 1;
        end
    end
end

% 儲存結果
save('correlation_ROIxLayer_max.mat', 'max_correlation');
save('correlation_ROIxLayer_mean.mat', 'mean_correlation');
fprintf('ROI X Layer 完成!\n');
fprintf('========== 所有相關性分析完成 ==========\n');

%% ========== 輔助函數 ==========

function img_stats = loadImageStats(subj_id, use_log)
    % 載入image statistics data
    % 輸入:
    %   subj_id: 受試者編號 (1-8)
    %   use_log: true = 載入 log-transformed 資料, false = 載入 raw 資料
    % 輸出:
    %   img_stats: cell array 包含 4 種統計特徵 {mean, contrast, skewness, slope}
    
    % 建構資料夾路徑
    addpath(sprintf('./img_stat/subj0%d', subj_id));
    
    % 根據參數決定檔名前綴
    if use_log
        prefix = 'log_';
    else
        prefix = '';
    end
    
    % 載入四種統計特徵
    img_mean = cell2mat(struct2cell(load(sprintf('%smean_subj0%d.mat', prefix, subj_id))));
    img_contrast = cell2mat(struct2cell(load(sprintf('%scontrast_subj0%d.mat', prefix, subj_id))));
    img_skw = cell2mat(struct2cell(load(sprintf('%sskewness_subj0%d.mat', prefix, subj_id))));
    img_slope = cell2mat(struct2cell(load(sprintf('%sfft_slope_subj0%d.mat', prefix, subj_id))));
    
    % 返回 cell array
    img_stats = {img_mean, img_contrast, img_skw, img_slope};
end

function feature_map = loadFeatureMap(subj_id, pool_id)
    % 載入降維後的 VGG-16 特徵圖
    % 輸入:
    %   subj_id: 受試者編號 (1-8)
    %   pool_id: pooling 層編號 (1-5)
    % 輸出:
    %   feature_map: 特徵圖矩陣 [n_samples x n_features]
    
    % 載入 .mat 檔案
    fmap_data = load(sprintf('feature_map_dimreduced_subj%d_pool%d.mat', subj_id, pool_id));
    
    % 提取特徵圖資料
    % 新版 NSD_VGG16.m 儲存為 'feature_map_data'
    if isfield(fmap_data, 'feature_map_data')
        feature_map = fmap_data.feature_map_data;
    else
        % 向後相容：如果是舊版格式，使用 cell2mat
        feature_map = cell2mat(struct2cell(fmap_data));
    end
end

function [fMRI_ROIs, mask] = loadfMRIData(subj_id, data_folder)
    % 載入 fMRI 資料並根據 ROI mask 分割
    % 輸入:
    %   subj_id: 受試者編號 (1-8)
    %   data_folder: fMRI 資料根目錄
    % 輸出:
    %   fMRI_ROIs: cell array 包含 7 個 ROI 的資料 {V1v, V1d, V2v, V2d, V3v, V3d, hV4}
    %   mask: 完整的 ROI mask (左右腦合併)
    
    % 載入左右腦的 ROI mask
    l_maskFILE = fullfile(data_folder, sprintf('subj0%d', subj_id), ...
                          'roi_masks', 'lh.prf-visualrois_challenge_space.npy');
    l_mask = transpose(double(readNPY(l_maskFILE)));
    
    r_maskFILE = fullfile(data_folder, sprintf('subj0%d', subj_id), ...
                          'roi_masks', 'rh.prf-visualrois_challenge_space.npy');
    r_mask = transpose(double(readNPY(r_maskFILE)));
    
    % 合併左右腦 mask
    mask = cat(2, l_mask, r_mask);
    
    % 載入左右腦的 fMRI 資料
    l_MRIfile = fullfile(data_folder, sprintf('subj0%d', subj_id), ...
                         'fmri', 'lh_fmri.npy');
    l_MRI = double(readNPY(l_MRIfile));
    
    r_MRIfile = fullfile(data_folder, sprintf('subj0%d', subj_id), ...
                         'fmri', 'rh_fmri.npy');
    r_MRI = double(readNPY(r_MRIfile));
    
    % 合併左右腦 fMRI 資料
    fMRI = cat(2, l_MRI, r_MRI);
    
    % 根據 mask 分割出各個 ROI
    % ROI 編號: 1-V1v, 2-V1d, 3-V2v, 4-V2d, 5-V3v, 6-V3d, 7-hV4
    fMRI_ROIs = cell(1, 7);
    for roi = 1:7
        fMRI_ROIs{roi} = fMRI(:, mask == roi);
    end
end