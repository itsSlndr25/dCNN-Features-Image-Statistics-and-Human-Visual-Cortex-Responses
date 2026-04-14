%% Image Statistics Calculator 
% 用途: 計算圖片的統計特徵 (mean, contrast, skewness, kurtosis, FFT slope)

% 輸入: Algonauts dataset 中的訓練圖片
% 輸出: 每個受試者的統計特徵 .mat 檔案 (raw 和 log-transformed 版本)
% 作者: I-HANG CHEN
% 日期: 2024 JUNE 14th

%% ========== 全域設定 ==========
% 資料路徑
DATA_FOLDER = '../data';
OUTPUT_FOLDER = '../data/img_stat';

% 受試者參數
NUM_SUBJECTS = 8;

% 統計特徵類型
COMPUTE_GLOBAL = true;   % 計算全域統計
COMPUTE_LOCAL = false;   % 計算局部統計 (設為 true 以啟用)
LOCAL_SEGMENTS = 16;     % 局部切割數量 (4x4 grid)

% 顏色空間轉換常數
SRGB_TO_CIE = [0.2126, 0.7152, 0.0722];  % sRGB to CIE XYZ 轉換矩陣
NEUTRAL_GRAY_EV = -2.44078736;           % 中性灰係數

fprintf('========== 開始計算圖片統計特徵 ==========\n');
fprintf('全域統計: %s\n', string(COMPUTE_GLOBAL));
fprintf('局部統計: %s\n', string(COMPUTE_LOCAL));

%% ========== 主迴圈：處理每個受試者 ==========
for subj_id = 1:NUM_SUBJECTS
    fprintf('\n處理受試者 %d/%d\n', subj_id, NUM_SUBJECTS);
    
    % 取得該受試者的圖片資料夾
    img_folder = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), 'images');
    img_files = dir(fullfile(img_folder, '*.png'));
    num_images = length(img_files);
    
    fprintf('  找到 %d 張圖片\n', num_images);
    
    %% === 計算全域統計 ===
    if COMPUTE_GLOBAL
        fprintf('  計算全域統計特徵...\n');
        global_stats = computeGlobalStatistics(img_folder, img_files, SRGB_TO_CIE, NEUTRAL_GRAY_EV);
        
        % 儲存全域統計
        output_path = fullfile(OUTPUT_FOLDER, sprintf('subj0%d', subj_id));
        saveStatistics(output_path, global_stats, subj_id, 'global');
        fprintf('  全域統計完成!\n');
    end
    
    %% === 計算局部統計 ===
    if COMPUTE_LOCAL
        fprintf('  計算局部統計特徵...\n');
        local_stats = computeLocalStatistics(img_folder, img_files, SRGB_TO_CIE, ...
                                             NEUTRAL_GRAY_EV, LOCAL_SEGMENTS);
        
        % 儲存局部統計
        output_path = fullfile(OUTPUT_FOLDER, sprintf('subj0%d', subj_id));
        saveStatistics(output_path, local_stats, subj_id, 'local');
        fprintf('  局部統計完成!\n');
    end
end

fprintf('\n========== 所有受試者處理完成 ==========\n');

%% ========== 輔助函數 ==========

function global_stats = computeGlobalStatistics(img_folder, img_files, sRGB_to_CIE, neutral_gray_EV)
    % 計算全域圖片統計特徵
    % 輸入:
    %   img_folder: 圖片資料夾路徑
    %   img_files: 圖片檔案列表 (dir 的輸出)
    %   sRGB_to_CIE: sRGB 到 CIE 的轉換係數
    %   neutral_gray_EV: 中性灰值
    % 輸出:
    %   global_stats: 結構，包含所有image statistics的 raw 和 log 版本
    
    num_images = length(img_files);
    
    % Allocate the space - Raw stat
    img_mean = zeros(1, num_images);
    img_contrast = zeros(1, num_images);
    img_skewness = zeros(1, num_images);
    img_kurtosis = zeros(1, num_images);
    img_fft_slope = zeros(1, num_images);
    
    % Allocate the space - Log stat
    log_img_mean = zeros(1, num_images);
    log_img_contrast = zeros(1, num_images);
    log_img_skewness = zeros(1, num_images);
    log_img_kurtosis = zeros(1, num_images);
    log_img_fft_slope = zeros(1, num_images);
    
    % 處理每張圖片
    for j = 1:num_images
        % 載入圖片
        filename = fullfile(img_folder, img_files(j).name);
        current_img = double(imread(filename));
        
        % 計算亮度圖 (raw 和 log 版本)
        [lumi, log_lumi] = computeLuminance(current_img, sRGB_to_CIE, neutral_gray_EV);
        
        % Flattened 成 1D 陣列
        [num_rows, num_cols] = size(lumi);
        lumi_array = reshape(lumi, [1, num_rows * num_cols]);
        log_lumi_array = reshape(log_lumi, [1, num_rows * num_cols]);
        
        % 計算統計量 - Raw 
        img_mean(j) = mean(lumi_array);
        img_contrast(j) = std(lumi_array);
        img_skewness(j) = skewness(lumi_array);
        img_kurtosis(j) = kurtosis(lumi_array);
        img_fft_slope(j) = getImageSpectSlope(lumi);
        
        % 計算統計量 - Log 
        log_img_mean(j) = mean(log_lumi_array);
        log_img_contrast(j) = std(log_lumi_array);
        log_img_skewness(j) = skewness(log_lumi_array);
        log_img_kurtosis(j) = kurtosis(log_lumi_array);
        log_img_fft_slope(j) = getImageSpectSlope(log_lumi);
    end
    
    % 將output包成新結構
    global_stats.raw.mean = img_mean;
    global_stats.raw.contrast = img_contrast;
    global_stats.raw.skewness = img_skewness;
    global_stats.raw.kurtosis = img_kurtosis;
    global_stats.raw.fft_slope = img_fft_slope;
    
    global_stats.log.mean = log_img_mean;
    global_stats.log.contrast = log_img_contrast;
    global_stats.log.skewness = log_img_skewness;
    global_stats.log.kurtosis = log_img_kurtosis;
    global_stats.log.fft_slope = log_img_fft_slope;
end

function local_stats = computeLocalStatistics(img_folder, img_files, sRGB_to_CIE, ...
                                               neutral_gray_EV, num_segments)
    % 計算local contrast
    % 輸入:
    %   img_folder: 圖片資料夾路徑
    %   img_files: 圖片檔案列表
    %   sRGB_to_CIE: sRGB 到 CIE 的轉換係數
    %   neutral_gray_EV: 中性灰曝光值
    %   num_segments: 局部分割數量 (預設 16 = 4x4 grid)
    % 輸出:
    %   local_stats: 結構，包含局部統計特徵
    
    num_images = length(img_files);
    
    % Allocate the space [num_segments x num_images]
    img_contrast = zeros(num_segments, num_images);
    log_img_contrast = zeros(num_segments, num_images);
    
    % 計算 grid 大小 (4x4)
    grid_size = sqrt(num_segments);  % 4
    
    % 處理每張圖片
    for j = 1:num_images
        % 載入圖片
        filename = fullfile(img_folder, img_files(j).name);
        current_img = double(imread(filename));
        
        % 計算亮度圖
        [lumi, log_lumi] = computeLuminance(current_img, sRGB_to_CIE, neutral_gray_EV);
        
        % 取得圖片尺寸
        [num_rows, num_cols] = size(lumi);
        row_stride = num_rows / grid_size;
        col_stride = num_cols / grid_size;
        
        % 遍歷每個局部區域
        segment_idx = 1;
        for row_step = 1:grid_size
            for col_step = 1:grid_size
                % 計算當前區域的範圍
                row_start = round(1 + row_stride * (row_step - 1));
                row_end = round(row_stride * row_step);
                col_start = round(1 + col_stride * (col_step - 1));
                col_end = round(col_stride * col_step);
                
                % 提取局部區域
                lumi_local = lumi(row_start:row_end, col_start:col_end);
                log_lumi_local = log_lumi(row_start:row_end, col_start:col_end);
                
                % 計算局部 contrast
                img_contrast(segment_idx, j) = std(lumi_local, 0, 'all');
                log_img_contrast(segment_idx, j) = std(log_lumi_local, 0, 'all');
                
                segment_idx = segment_idx + 1;
            end
        end
    end
    
    % 將結果打包成結構
    local_stats.raw.contrast = img_contrast;
    local_stats.log.contrast = log_img_contrast;
end

function [lumi, log_lumi] = computeLuminance(img, sRGB_to_CIE, neutral_gray_EV)
    % 計算圖片的亮度圖 (luminance map)
    % 輸入:
    %   img: RGB 圖片 [H x W x 3]，範圍 0-255
    %   sRGB_to_CIE: sRGB 到 CIE 的轉換係數 [1x3]
    %   neutral_gray_EV: 中性灰曝光值係數
    % 輸出:
    %   lumi: 線性亮度圖 [H x W]
    %   log_lumi: log-transformed 亮度圖 [H x W]
    %
    % Note: 使用此轉換而非 MATLAB 原有的 rgb2gray，目的是確保視覺實驗的精確性
    
    % Normalize到 0-1
    img_normalized = img / 255;
    
    % Allocate線性空間圖片
    linear_img = zeros(size(img));
    
    % sRGB gamma correction (向量化運算)
    % 根據 sRGB : 
    %   - low value: linear = sRGB / 12.92
    %   - high value: linear = ((sRGB + 0.055) / 1.055) ^ 2.4
    low_mask = img_normalized <= 0.04045;
    high_mask = img_normalized > 0.04045;
    
    linear_img(low_mask) = img_normalized(low_mask) / 12.92;
    linear_img(high_mask) = ((img_normalized(high_mask) + 0.055) / 1.055) .^ 2.4;
    
    % 計算亮度 (weighted sum of RGB channels)
    % 使用 CIE 1931 standard: Y = 0.2126*R + 0.7152*G + 0.0722*B
    lumi = sum(linear_img .* reshape(sRGB_to_CIE, [1, 1, 3]), 3);
    
    % 計算 log-transformed 亮度 (曝光值 EV)
    % 避免 log(0) 的問題，加上極小值eps
    lumi_safe = max(lumi, eps);
    log_lumi = log2(lumi_safe) - neutral_gray_EV;
end

function saveStatistics(output_path, stats, subj_id, stat_type)
    % 儲存統計特徵到 .mat 檔案
    % 輸入:
    %   output_path: 輸出資料夾路徑
    %   stats: 統計特徵結構 (包含 raw 和 log 子結構)
    %   subj_id: 受試者編號
    %   stat_type: 'global' 或 'local'
    
    % 確保輸出資料夾存在
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    % 根據類型決定檔名前綴
    if strcmp(stat_type, 'local')
        prefix = 'local_';
    else
        prefix = '';
    end
    
    % 取得image statistics名稱
    stat_names = fieldnames(stats.raw);
    
    % 儲存每個image statistics
    for i = 1:length(stat_names)
        stat_name = stat_names{i};
        
        % 儲存 raw 版本
        data = stats.raw.(stat_name);
        filename = fullfile(output_path, sprintf('%s%s_subj0%d.mat', prefix, stat_name, subj_id));
        
        % 動態變數名稱 (為了與原始格式相容)
        if strcmp(stat_type, 'local')
            var_name = 'img_contrast';  % local 版本只有 contrast
        else
            var_name = ['img_' stat_name];
        end
        eval([var_name ' = data;']);
        save(filename, var_name);
        
        % 儲存 log 版本
        data = stats.log.(stat_name);
        filename = fullfile(output_path, sprintf('%slog_%s_subj0%d.mat', prefix, stat_name, subj_id));
        
        if strcmp(stat_type, 'local')
            var_name = 'log_img_contrast';
        else
            var_name = ['log_img_' stat_name];
        end
        eval([var_name ' = data;']);
        save(filename, var_name);
    end
end