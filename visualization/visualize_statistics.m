%% Image Statistics Distribution Visualization
% 用途: 視覺化Image Statistics的分佈直方圖
% 輸入: img_stats.m 產生的統計特徵 .mat 檔案
% 輸出: 各Image Statistics的分佈直方圖 (PNG 檔案)
% 作者: I-HANG CHEN
% 日期: 2024 JULY 20st

%% ========== 全域設定 ==========
% 資料路徑
DATA_FOLDER = '../data/img_stat';  % 統計特徵資料夾
OUTPUT_FOLDER = '../results';  % 圖表輸出資料夾

% 受試者參數
NUM_SUBJECTS = 8;

% Image Statistics類型
STAT_TYPES = {'mean', 'log_contrast', 'skewness', 'log_fft_slope'};
STAT_NAMES = {'Mean', 'Log Contrast', 'Skewness', 'Log FFT Slope'};
OUTPUT_NAMES = {'mean', 'contrast', 'skewness', 'slope'};

% 視覺化參數
NUM_BINS = 250;           % 直方圖的 bin 數量
BAR_COLOR = [70, 192, 179] / 255;  % 長條顏色（青綠色）

% 確保輸出資料夾存在
if ~exist(OUTPUT_FOLDER, 'dir')
    mkdir(OUTPUT_FOLDER);
end

fprintf('========== Image Statistics分佈視覺化 ==========\n');
fprintf('受試者數量: %d\n', NUM_SUBJECTS);
fprintf('Statistics數量: %d\n', length(STAT_TYPES));
fprintf('輸出資料夾: %s\n\n', OUTPUT_FOLDER);

%% ========== 載入並合併所有受試者的Image Statistics資料 ==========
fprintf('載入Image Statistics資料...\n');

% 初始化資料儲存
all_stats = struct();
for stat_idx = 1:length(STAT_TYPES)
    all_stats.(OUTPUT_NAMES{stat_idx}) = [];
end

% 載入每個受試者的資料
total_images = 0;
for subj_id = 1:NUM_SUBJECTS
    fprintf('  載入受試者 %d/%d...', subj_id, NUM_SUBJECTS);
    
    % 構建資料夾路徑
    subj_folder = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id));
    
    % 載入各統計特徵
    for stat_idx = 1:length(STAT_TYPES)
        stat_type = STAT_TYPES{stat_idx};
        output_name = OUTPUT_NAMES{stat_idx};
        
        % 構建檔案名稱
        filename = fullfile(subj_folder, sprintf('%s_subj0%d.mat', stat_type, subj_id));
        
        % 載入資料
        if exist(filename, 'file')
            loaded_data = load(filename);
            
            % 提取變數（假設只有一個變數）
            field_names = fieldnames(loaded_data);
            stat_data = loaded_data.(field_names{1});
            
            % 合併到總資料
            all_stats.(output_name) = [all_stats.(output_name), stat_data];
        else
            warning('檔案不存在: %s', filename);
        end
    end
    
    fprintf(' 完成\n');
end

% 計算總圖片數
total_images = length(all_stats.(OUTPUT_NAMES{1}));
fprintf('\n總圖片數: %d\n\n', total_images);

%% ========== 生成分佈圖 ==========
fprintf('生成分佈圖...\n');

for stat_idx = 1:length(STAT_TYPES)
    output_name = OUTPUT_NAMES{stat_idx};
    stat_name = STAT_NAMES{stat_idx};
    
    fprintf('  繪製 %s 分佈...', stat_name);
    
    % 取得資料
    stat_data = all_stats.(output_name);
    
    % 移除極端值（根據需求調整）
    % stat_data = removeOutliers(stat_data);
    
    % 建立新圖表
    figure('Position', [100, 100, 800, 600]);
    
    % 繪製直方圖
    h = histogram(stat_data, NUM_BINS, ...
        'FaceColor', BAR_COLOR, ...
        'EdgeColor', BAR_COLOR, ...
        'FaceAlpha', 0.7);
    
    % 設定標題和標籤
    title(sprintf('%s Distribution (N=%d)', stat_name, total_images), ...
        'FontSize', 14, 'FontWeight', 'bold');
    xlabel(stat_name, 'FontSize', 12);
    ylabel('Frequency', 'FontSize', 12);
    
    % 設定格線
    grid on;
    
    % 加上統計資訊
    stat_mean = mean(stat_data);
    stat_std = std(stat_data);
    stat_median = median(stat_data);
    
    % 在圖上顯示統計資訊
    text_str = sprintf('Mean: %.3f\nStd: %.3f\nMedian: %.3f', ...
        stat_mean, stat_std, stat_median);
    text(0.7, 0.9, text_str, 'Units', 'normalized', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black', ...
        'FontSize', 10);
    
    % 儲存圖表
    output_filename = fullfile(OUTPUT_FOLDER, ...
        sprintf('%s_distribution.png', output_name));
    saveas(gcf, output_filename);
    
    % 關閉圖表
    close(gcf);
    
    fprintf(' 完成\n');
end

fprintf('\n========================================\n');
fprintf('所有分佈圖生成完成!\n');
fprintf('========================================\n');
fprintf('輸出位置: %s\n', OUTPUT_FOLDER);
fprintf('總共產生 %d 張圖表\n\n', length(STAT_TYPES));

%% ========== (可選) 計算並儲存分組閾值 ==========
fprintf('計算Image Statistics分組閾值...\n');

% 計算閾值用於將圖片分為高/低或高/中/低組
thresholds = struct();

% Mean: 高/低 (中位數分割)
thresholds.mean_cut = median(all_stats.mean);

% Contrast: 高/低 (中位數分割)
thresholds.contrast_cut = median(all_stats.contrast);

% Skewness: 高/中/低 (33rd, 67th 百分位數)
thresholds.skewness_cut = prctile(all_stats.skewness, [33, 67]);

% Slope: 高/中/低 (33rd, 67th 百分位數)
thresholds.slope_cut = prctile(all_stats.slope, [33, 67]);

% 顯示閾值
fprintf('\n統計特徵分組閾值:\n');
fprintf('  Mean (median): %.4f\n', thresholds.mean_cut);
fprintf('  Contrast (median): %.4f\n', thresholds.contrast_cut);
fprintf('  Skewness (33rd, 67th): [%.4f, %.4f]\n', ...
    thresholds.skewness_cut(1), thresholds.skewness_cut(2));
fprintf('  Slope (33rd, 67th): [%.4f, %.4f]\n', ...
    thresholds.slope_cut(1), thresholds.slope_cut(2));

% 儲存閾值
threshold_filename = fullfile(OUTPUT_FOLDER, 'stat_thresholds.mat');
save(threshold_filename, 'thresholds');
fprintf('\n閾值已儲存至: %s\n\n', threshold_filename);

%% ========== 輔助函數 ==========

function clean_data = removeOutliers(data, num_std)
% REMOVEOUTLIERS - 移除極端值
%
% 輸入:
%   data     - 原始資料陣列
%   num_std  - (可選) 標準差倍數，預設為 3
% 輸出:
%   clean_data - 移除極端值後的資料

    if nargin < 2
        num_std = 3;
    end
    
    data_mean = mean(data);
    data_std = std(data);
    
    % 保留在 mean ± num_std*std 範圍內的資料
    lower_bound = data_mean - num_std * data_std;
    upper_bound = data_mean + num_std * data_std;
    
    clean_data = data(data >= lower_bound & data <= upper_bound);
    
    num_removed = length(data) - length(clean_data);
    if num_removed > 0
        fprintf('  移除了 %d 個極端值 (%.2f%%)\n', ...
            num_removed, 100 * num_removed / length(data));
    end
end