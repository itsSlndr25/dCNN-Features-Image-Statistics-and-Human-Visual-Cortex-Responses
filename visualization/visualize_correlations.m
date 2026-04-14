%% Correlation Analysis Visualization - Bar Charts
% 用途: 將相關性分析結果視覺化為長條圖
% 輸入: correlation_do_all.m 產生的 .mat 檔案
% 輸出: 各種相關性分析的長條圖 (PNG 檔案)
% 作者: I-HANG CHEN
% 日期: 2024 JULY 20st

%% ========== 全域設定 ==========
% 視覺化類型
PLOT_TYPE = 'max';  % 'max' 或 'mean' - 選擇要視覺化最大值或平均值

% 輸出設定
OUTPUT_FOLDER = '../results/figures';  % 圖表輸出資料夾
INPUT_FOLDER = '../results';           % 圖表輸入資料夾
FIG_FORMAT = 'png';                    % 圖片格式
FIG_DPI = 300;                         % 圖片解析度

% 確保輸出資料夾存在
if ~exist(OUTPUT_FOLDER, 'dir')
    mkdir(OUTPUT_FOLDER);
end

% 顏色配置（用於不同的統計特徵或層級）
COLOR_SCHEME = struct();
COLOR_SCHEME.stat1 = [242, 60, 87] / 255;    % 紅色 - mean
COLOR_SCHEME.stat2 = [165, 43, 122] / 255;   % 紫色 - contrast
COLOR_SCHEME.stat3 = [81, 83, 117] / 255;    % 灰藍 - skewness
COLOR_SCHEME.stat4 = [35, 141, 95] / 255;    % 綠色 - fft slope
COLOR_SCHEME.layer1 = [0.2010, 0.5450, 0.6330];  % pool1
COLOR_SCHEME.layer2 = [0.4940, 0.1840, 0.5560];  % pool2
COLOR_SCHEME.layer3 = [0.9290, 0.6940, 0.1250];  % pool3
COLOR_SCHEME.layer4 = [0.7500, 0.0950, 0.0980];  % pool4
COLOR_SCHEME.layer5 = [0.3660, 0.6740, 0.4080];  % pool5

fprintf('========== 開始生成相關性分析圖表 ==========\n');
fprintf('視覺化類型: %s\n', PLOT_TYPE);
fprintf('輸出資料夾: %s\n\n', OUTPUT_FOLDER);

%% ========== Section 1: ROI x Stat (Log-transformed) ==========
fprintf('生成圖表 1/5: ROI x Stat (Log)...\n');

plotCorrelationBarChart(...
    INPUT_FOLDER + 'correlation_ROIxStat_max.mat', ...
    INPUT_FOLDER + 'correlation_ROIxStat_mean.mat', ...
    PLOT_TYPE, ...
    {'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'}, ...
    {'mean', 'contrast', 'skewness', 'fft slope'}, ...
    [COLOR_SCHEME.stat1; COLOR_SCHEME.stat2; COLOR_SCHEME.stat3; COLOR_SCHEME.stat4], ...
    'ROI x Stat (Log-transformed)', ...
    'ROI', ...
    'Correlation Coefficient', ...
    fullfile(OUTPUT_FOLDER, sprintf('ROIxStat_%s.%s', PLOT_TYPE, FIG_FORMAT)));

%% ========== Section 2: ROI x Stat (Raw) ==========
fprintf('生成圖表 2/5: ROI x Stat (Raw)...\n');

plotCorrelationBarChart(...
    INPUT_FOLDER + 'correlation_rawROIxStat_max.mat', ...
    INPUT_FOLDER + 'correlation_rawROIxStat_mean.mat', ...
    PLOT_TYPE, ...
    {'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'}, ...
    {'mean', 'contrast', 'skewness', 'fft slope'}, ...
    [COLOR_SCHEME.stat1; COLOR_SCHEME.stat2; COLOR_SCHEME.stat3; COLOR_SCHEME.stat4], ...
    'ROI x Stat (Raw)', ...
    'ROI', ...
    'Correlation Coefficient', ...
    fullfile(OUTPUT_FOLDER, sprintf('raw_ROIxStat_%s.%s', PLOT_TYPE, FIG_FORMAT)));

%% ========== Section 3: ROI x Layer ==========
fprintf('生成圖表 3/5: ROI x Layer...\n');

plotCorrelationBarChart(...
    INPUT_FOLDER + 'correlation_ROIxLayer_max.mat', ...
    INPUT_FOLDER + 'correlation_ROIxLayer_mean.mat', ...
    PLOT_TYPE, ...
    {'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'}, ...
    {'pool1', 'pool2', 'pool3', 'pool4', 'pool5'}, ...
    [COLOR_SCHEME.layer1; COLOR_SCHEME.layer2; COLOR_SCHEME.layer3; ...
     COLOR_SCHEME.layer4; COLOR_SCHEME.layer5], ...
    'ROI x Layer', ...
    'ROI', ...
    'Correlation Coefficient', ...
    fullfile(OUTPUT_FOLDER, sprintf('ROIxLayer_%s.%s', PLOT_TYPE, FIG_FORMAT)));

%% ========== Section 4: Layer x Stat (Log-transformed) ==========
fprintf('生成圖表 4/5: Layer x Stat (Log)...\n');

plotCorrelationBarChart(...
    INPUT_FOLDER + 'correlation_LayerxStat_max.mat', ...
    INPUT_FOLDER + 'correlation_LayerxStat_mean.mat', ...
    PLOT_TYPE, ...
    {'pool1', 'pool2', 'pool3', 'pool4', 'pool5'}, ...
    {'mean', 'contrast', 'skewness', 'fft slope'}, ...
    [COLOR_SCHEME.layer1; COLOR_SCHEME.layer2; COLOR_SCHEME.layer3; ...
     COLOR_SCHEME.layer4; COLOR_SCHEME.layer5], ...
    'Layer x Stat (Log-transformed)', ...
    'Layer', ...
    'Correlation Coefficient', ...
    fullfile(OUTPUT_FOLDER, sprintf('LayerxStat_%s.%s', PLOT_TYPE, FIG_FORMAT)));

%% ========== Section 5: Layer x Stat (Raw) ==========
fprintf('生成圖表 5/5: Layer x Stat (Raw)...\n');

plotCorrelationBarChart(...
    INPUT_FOLDER + 'correlation_rawLayerxStat_max.mat', ...
    INPUT_FOLDER + 'correlation_rawLayerxStat_mean.mat', ...
    PLOT_TYPE, ...
    {'pool1', 'pool2', 'pool3', 'pool4', 'pool5'}, ...
    {'mean', 'contrast', 'skewness', 'fft slope'}, ...
    [COLOR_SCHEME.layer1; COLOR_SCHEME.layer2; COLOR_SCHEME.layer3; ...
     COLOR_SCHEME.layer4; COLOR_SCHEME.layer5], ...
    'Layer x Stat (Raw)', ...
    'Layer', ...
    'Correlation Coefficient', ...
    fullfile(OUTPUT_FOLDER, sprintf('raw_LayerxStat_%s.%s', PLOT_TYPE, FIG_FORMAT)));

fprintf('\n========================================\n');
fprintf('所有圖表生成完成!\n');
fprintf('========================================\n');
fprintf('輸出位置: %s\n', OUTPUT_FOLDER);
fprintf('總共產生 %d 張圖表\n\n', 5);

%% ========== 輔助函數 ==========

function plotCorrelationBarChart(max_file, mean_file, plot_type, ...
    x_categories, legend_labels, colors, title_text, xlabel_text, ...
    ylabel_text, output_filename)
% PLOTCORRELATIONBARCHART - 繪製相關性分析的長條圖
%
% 輸入參數:
%   max_file        - correlation (max)資料檔案路徑
%   mean_file       - correlation (mean)資料檔案路徑
%   plot_type       - 'max' 或 'mean'
%   x_categories    - X 軸類別標籤 (cell array)
%   legend_labels   - 圖例標籤 (cell array)
%   colors          - 長條顏色矩陣 [n_bars x 3]
%   title_text      - 圖表標題
%   xlabel_text     - X 軸標籤
%   ylabel_text     - Y 軸標籤
%   output_filename - 輸出檔案完整路徑
%
% 輸出:
%   儲存圖表到指定檔案

    % 載入資料
    max_data = loadCorrelationData(max_file);
    mean_data = loadCorrelationData(mean_file);
    
    % 根據 plot_type 選擇資料並計算跨受試者平均
    switch lower(plot_type)
        case 'max'
            correlation_avg = mean(max_data, 3);  % 第3維是受試者
        case 'mean'
            correlation_avg = mean(mean_data, 3);
        otherwise
            error('plot_type 必須是 "max" 或 "mean"');
    end
    
    % 轉置資料使其符合 bar() 的格式
    % bar() 期望 [n_categories x n_groups] 的格式
    Y = correlation_avg';
    
    % 建立新圖表
    figure('Position', [100, 100, 800, 600]);
    
    % 建立 categorical X 軸並指定順序
    X = categorical(x_categories);
    X = reordercats(X, x_categories);
    
    % 繪製長條圖
    b = bar(X, Y);
    
    % 設定每個長條的顏色
    for i = 1:length(b)
        if i <= size(colors, 1)
            b(i).FaceColor = colors(i, :);
            b(i).EdgeColor = colors(i, :);
        end
    end
    
    % 設定標題和標籤
    title(title_text, 'FontSize', 14, 'FontWeight', 'bold');
    xlabel(xlabel_text, 'FontSize', 12);
    ylabel(ylabel_text, 'FontSize', 12);
    
    % 設定圖例
    if length(legend_labels) == length(b)
        legend(legend_labels, 'Location', 'best', 'FontSize', 10);
    end
    
    % 設定格線
    grid on;
    grid minor;
    
    % 設定 Y 軸範圍（確保從 0 開始）
    ylim([0, max(Y(:)) * 1.1]);
    
    % 儲存圖表
    saveas(gcf, output_filename);
    
    % 關閉圖表（釋放記憶體）
    close(gcf);
end

function data = loadCorrelationData(filename)
% LOADCORRELATIONDATA - 載入相關性資料檔案
%
% 輸入:
%   filename - .mat 檔案路徑
% 輸出:
%   data - 相關性資料矩陣
%
% 注意: 此函數與 correlation_do_all.m 優化後的變數命名相容

    % 載入 .mat 檔案
    loaded = load(filename);
    
    % 嘗試不同的變數名稱（向後相容）
    if isfield(loaded, 'max_correlation')
        data = loaded.max_correlation;
    elseif isfield(loaded, 'mean_correlation')
        data = loaded.mean_correlation;
    elseif isfield(loaded, 'maxi')
        % 向後相容舊版本
        data = loaded.maxi;
    elseif isfield(loaded, 'Mu')
        % 向後相容舊版本
        data = loaded.Mu;
    else
        % 如果都找不到，使用第一個變數
        field_names = fieldnames(loaded);
        if ~isempty(field_names)
            data = loaded.(field_names{1});
        else
            error('無法從檔案 %s 載入資料', filename);
        end
    end
end