%% fMRI ROI Masking - 優化版本
% 用途: 從 fMRI 資料中提取各個視覺腦區 (Region of Interest) 的 voxel 資料
% 輸入: Algonauts dataset 的 fMRI 資料和 ROI mask
% 輸出: 每個受試者每個腦區的 voxel 時間序列
%       - subj01_V1.mat, subj01_V2.mat, etc.
% Region of Interest定義:
%   - V1: mask == 1 (V1v) | mask == 2 (V1d)
%   - V2: mask == 3 (V2v) | mask == 4 (V2d)
%   - V3: mask == 5 (V3v) | mask == 6 (V3d)
%   - V4: mask == 7 (hV4)
% 作者: I-HANG CHEN
% 日期: 2024 MAY 29th

%% ========== 全域設定 ==========
% 資料路徑 (當時環境)
DATA_FOLDER = './fMRI_Datasets/algonauts/data'; % actual path required
OUTPUT_FOLDER = '../data/ROIs';  % 輸出資料夾

% 確保輸出資料夾存在
if ~exist(OUTPUT_FOLDER, 'dir')
    mkdir(OUTPUT_FOLDER);
end

% 添加 NPY 讀取工具路徑
addpath('../utils/readNPY.m');

% 受試者與 ROI 參數
NUM_SUBJECTS = 8;
NUM_ROIS = 7;  
ROI_NAMES = {'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4'};

% ROI mask 值定義
% V1: V1v(1) + V1d(2), V2: V2v(3) + V2d(4), V3: V3v(5) + V3d(6), V4: hV4(7)
ROI_MASK_VALUES = {
    [1], [2],    % V1v & V1d
    [3], [4],    % V2v & V2d
    [5], [6],    % V3v & V2d
    [7]        % hV4
};

fprintf('========== fMRI ROI 資料提取 ==========\n');
fprintf('受試者數: %d\n', NUM_SUBJECTS);
fprintf('ROI 數: %d (%s)\n', NUM_ROIS, strjoin(ROI_NAMES, ', '));
fprintf('輸出路徑: %s\n\n', OUTPUT_FOLDER);

%% ========== 主迴圈: 處理每個受試者 ==========
for subj_id = 1:NUM_SUBJECTS
    fprintf('處理受試者 %d/%d\n', subj_id, NUM_SUBJECTS);
    
    %% === 載入 ROI Mask ===
    % 左腦 mask
    lh_mask_file = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), ...
        'roi_masks', 'lh.prf-visualrois_challenge_space.npy');
    lh_mask = readNPY(lh_mask_file);
    lh_mask = double(lh_mask(:)');  % 轉置並攤平
    
    % 右腦 mask
    rh_mask_file = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), ...
        'roi_masks', 'rh.prf-visualrois_challenge_space.npy');
    rh_mask = readNPY(rh_mask_file);
    rh_mask = double(rh_mask(:)');  % 轉置並攤平
    
    % 合併左右腦 mask
    full_mask = [lh_mask, rh_mask];
    
    fprintf('  Mask 載入完成 (總 voxels: %d)\n', length(full_mask));
    
    %% === 載入 fMRI 資料 ===
    % 左腦 fMRI
    lh_fmri_file = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), ...
        'fmri', 'lh_fmri.npy');
    lh_fmri = double(readNPY(lh_fmri_file));
    
    % 右腦 fMRI
    rh_fmri_file = fullfile(DATA_FOLDER, sprintf('subj0%d', subj_id), ...
        'fmri', 'rh_fmri.npy');
    rh_fmri = double(readNPY(rh_fmri_file));
    
    % 合併左右腦 fMRI [n_timepoints x n_voxels]
    full_fmri = [lh_fmri, rh_fmri];
    
    fprintf('  fMRI 載入完成 (時間點: %d, voxels: %d)\n', ...
        size(full_fmri, 1), size(full_fmri, 2));
    
    %% === 提取並儲存各 ROI 資料 ===
    for roi_id = 1:NUM_ROIS
        roi_name = ROI_NAMES{roi_id};
        mask_values = ROI_MASK_VALUES{roi_id};
        
        % 建立 ROI 的 boolean mask
        roi_mask = ismember(full_mask, mask_values);
        
        % 提取該 ROI 的所有 voxels
        roi_data = full_fmri(:, roi_mask);
        num_voxels = size(roi_data, 2);
        
        % 儲存 ROI 資料
        % 為了相容性，使用動態變數名
        output_filename = fullfile(OUTPUT_FOLDER, ...
            sprintf('subj0%d_%s.mat', subj_id, roi_name));
        
        % 動態變數命名: MRI_V1, MRI_V2, etc.
        var_name = sprintf('MRI_%s', roi_name);
        eval([var_name ' = roi_data;']);
        save(output_filename, var_name);
        
        fprintf('  %s: %d voxels 已儲存\n', roi_name, num_voxels);
    end
    
    fprintf('受試者 %d 完成!\n\n', subj_id);
end

fprintf('========================================\n');
fprintf('所有受試者處理完成!\n');
fprintf('========================================\n');
fprintf('總共產生 %d 個 ROI 檔案\n', NUM_SUBJECTS * NUM_ROIS);
fprintf('輸出位置: %s\n\n', OUTPUT_FOLDER);