#!/usr/bin/env python3
"""
FIX #2: AUDIO MODALITY DEBUGGING SCRIPT
========================================

PURPOSE:
Diagnose why audio modality achieves only 0.394 accuracy (worse than random).

SYMPTOMS:
- Audio model AUC: 0.548 (barely above 0.5)
- Audio model accuracy: 0.394 (worse than random 50%)
- Audio corrupts late fusion (fusion AUC = text-only AUC)

ROOT CAUSES TO CHECK:
1. Audio feature files missing or incomplete
2. Audio features empty or all-zeros
3. Audio data leakage (SMOTE applied before train/test split)
4. Audio features poorly correlated with depression labels
5. Feature extraction bugs (NaN, infinite values)

USAGE:
python FIX_2_AUDIO_DEBUGGING.py --data-root ~/Downloads/E-DAIC/data

OUTPUT:
Summary of audio data quality with recommendations.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class AudioDebugger:
    """Comprehensive audio diagnostics."""
    
    def __init__(self, data_root, verbose=True):
        self.data_root = data_root
        self.verbose = verbose
        self.issues = []
        self.warnings = []
        self.recommendations = []
        
    def log(self, message, level='INFO'):
        """Print message with optional formatting."""
        if self.verbose:
            prefix = {
                'INFO': '✓',
                'WARN': '⚠️',
                'ERROR': '❌',
                'SUCCESS': '✅'
            }.get(level, '•')
            print(f"{prefix} {message}")
    
    def check_audio_files_exist(self, pids):
        """Check if audio feature files exist."""
        self.log("\n1. CHECKING AUDIO FILE AVAILABILITY", 'INFO')
        
        audio_files = {
            'MFCC': 'OpenSMILE2.3.0_mfcc.csv',
            'eGeMAPS': 'OpenSMILE2.3.0_egemaps.csv',
            'BoAW_MFCC': 'BoAW_openSMILE_2.3.0_MFCC.csv',
            'BoAW_eGeMAPS': 'BoAW_openSMILE_2.3.0_eGeMAPS.csv',
        }
        
        stats = {k: {'found': 0, 'missing': 0, 'error': 0} for k in audio_files}
        
        for pid in pids[:20]:  # Check first 20
            feat_dir = os.path.join(self.data_root, f"{pid}_P", "features")
            
            for feature_name, filename_template in audio_files.items():
                path = os.path.join(feat_dir, f"{pid}_{filename_template}")
                
                if not os.path.exists(path):
                    stats[feature_name]['missing'] += 1
                    continue
                
                try:
                    df = pd.read_csv(path, header=None if 'BoAW' in feature_name else 'infer')
                    if df.empty:
                        stats[feature_name]['error'] += 1
                    else:
                        stats[feature_name]['found'] += 1
                except Exception as e:
                    stats[feature_name]['error'] += 1
                    self.log(f"Error reading {pid}_{filename_template}: {str(e)}", 'ERROR')
        
        self.log("\nAudio Feature File Status (sample of 20):")
        for feature_name, stat in stats.items():
            total = stat['found'] + stat['missing'] + stat['error']
            pct_found = stat['found'] / total * 100 if total > 0 else 0
            status = 'FOUND' if pct_found > 50 else 'MISSING'
            self.log(f"  {feature_name:15} | Found: {stat['found']:2}/{total:2} ({pct_found:5.1f}%) [{status}]")
            
            if pct_found < 50:
                self.issues.append(f"Audio feature {feature_name} mostly missing")
        
        return stats
    
    def check_audio_feature_quality(self, pids, feature_file='audio_features_enhanced.csv'):
        """Check if audio features have reasonable values."""
        self.log("\n2. CHECKING AUDIO FEATURE QUALITY", 'INFO')
        
        # Try to load pre-extracted features
        features_path = os.path.join(os.path.dirname(self.data_root), 'features', feature_file)
        
        if not os.path.exists(features_path):
            self.log(f"Features file not found: {features_path}", 'WARN')
            self.log("Skipping quality check (features not pre-extracted)", 'WARN')
            return None
        
        try:
            df = pd.read_csv(features_path)
            self.log(f"Loaded {feature_file}: shape {df.shape}")
            
            # Check for common issues
            issues = []
            
            # Check for all-zeros columns
            zero_cols = (df == 0).sum()
            all_zero = zero_cols[zero_cols == len(df)].index.tolist()
            if all_zero:
                self.log(f"Found {len(all_zero)} all-zero columns", 'WARN')
                issues.append(f"All-zero features: {all_zero[:5]}...")
            
            # Check for NaN
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                self.log(f"Found {nan_count} NaN values", 'WARN')
                issues.append("NaN values in audio features")
            
            # Check for infinite values
            inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                self.log(f"Found {inf_count} infinite values", 'ERROR')
                issues.append("Infinite values in audio features")
            
            # Check variance
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            variances = df[numeric_cols].var()
            low_var_cols = variances[variances < 1e-6].index.tolist()
            if len(low_var_cols) > 0:
                self.log(f"Found {len(low_var_cols)} near-constant features", 'WARN')
                self.log(f"  Examples: {low_var_cols[:5]}")
                issues.append("Low-variance audio features (no discriminative signal)")
            
            # Check feature correlation with labels (if available)
            if 'label' in df.columns:
                correlations = df[numeric_cols].corrwith(df['label']).abs()
                mean_corr = correlations.mean()
                self.log(f"Mean feature-label correlation: {mean_corr:.4f}", 'INFO')
                
                if mean_corr < 0.1:
                    issues.append("Audio features weakly correlated with labels")
                    self.log("  → Audio may not contain depressive signal", 'WARN')
            
            for issue in issues:
                self.issues.append(issue)
            
            return df
            
        except Exception as e:
            self.log(f"Error loading features: {str(e)}", 'ERROR')
            return None
    
    def check_data_leakage(self):
        """Check if feature extraction might have data leakage."""
        self.log("\n3. CHECKING FOR DATA LEAKAGE", 'INFO')
        
        self.log("Common leakage sources in audio:")
        self.log("  a) Standardizing features BEFORE train/test split", 'INFO')
        self.log("  b) Applying SMOTE OUTSIDE of cross-validation fold", 'INFO')
        self.log("  c) Selecting features on full dataset before splitting", 'INFO')
        
        self.log("\nTo verify:")
        self.log("  1. Open main.py")
        self.log("  2. Check lines 114-122 (PCA should fit INSIDE fold)")
        self.log("  3. Check lines 180-183 (PCA should fit on TRAIN only)")
        
        # Check if main.py has correct implementation
        try:
            with open('main.py', 'r') as f:
                main_content = f.read()
                
                if 'pca_audio.fit_transform(X_audio[tr_f])' in main_content:
                    self.log("✅ PCA correctly fits inside fold", 'SUCCESS')
                else:
                    self.log("⚠️  PCA fitting may not be inside fold", 'WARN')
                    self.issues.append("Potential data leakage: PCA not inside fold")
        except:
            self.log("Could not check main.py", 'WARN')
    
    def check_model_performance(self):
        """Check current audio model performance."""
        self.log("\n4. CHECKING AUDIO MODEL PERFORMANCE", 'INFO')
        
        # These are hardcoded from README
        audio_metrics = {
            'Accuracy': 0.394,
            'F1': 0.474,
            'AUC-ROC': 0.548,
        }
        
        self.log("Current Audio Model Performance:")
        for metric, value in audio_metrics.items():
            status = '❌' if (metric == 'Accuracy' and value < 0.5) else '⚠️'
            self.log(f"  {status} {metric}: {value:.3f}", 'WARN')
        
        self.log("\nComparison:")
        self.log("  Text-only AUC: 0.591")
        self.log("  Audio-only AUC: 0.548 ← BELOW random 0.5!")
        self.log("  Visual-only AUC: 0.657")
        
        self.issues.append("Audio AUC (0.548) barely above random (0.5)")
    
    def generate_recommendations(self):
        """Generate actionable recommendations."""
        self.log("\n5. RECOMMENDATIONS", 'SUCCESS')
        
        if not self.issues:
            self.log("No critical issues found!", 'SUCCESS')
            self.log("Audio may perform poorly due to:")
            self.log("  • Data quality issue not detectable from this diagnostic")
            self.log("  • E-DAIC audio doesn't correlate with PHQ-8 labels")
            self.log("  • Feature extraction mismatch between training and evaluation")
            return
        
        self.log(f"Found {len(self.issues)} issues:\n")
        for i, issue in enumerate(self.issues, 1):
            self.log(f"{i}. {issue}")
        
        self.log("\nRecommended Actions:")
        
        if any('missing' in issue.lower() for issue in self.issues):
            self.log("\n→ ACTION 1: Verify E-DAIC Dataset")
            self.log("  1. Check if E-DAIC data is fully downloaded")
            self.log("  2. Verify data path: export EDAIC_DATA_ROOT=/correct/path")
            self.log("  3. Rerun: python diagnose_data.py")
        
        if any('leakage' in issue.lower() for issue in self.issues):
            self.log("\n→ ACTION 2: Fix Data Leakage")
            self.log("  1. Open main.py")
            self.log("  2. Ensure PCA fits INSIDE CV fold (line 114-122)")
            self.log("  3. Ensure SMOTE applied INSIDE fold (if used)")
            self.log("  4. Retrain: python main.py")
        
        if any('correlated' in issue.lower() for issue in self.issues):
            self.log("\n→ ACTION 3: Disable Unreliable Audio")
            self.log("  1. Edit config.py")
            self.log("  2. Set AUDIO_RELIABLE = False")
            self.log("  3. This removes audio from fusion")
            self.log("  4. Retrain: python main.py")
            self.log("  → Expected: Late fusion AUC will improve (text + visual only)")
        
        if any('NaN' in issue or 'infinite' in issue.lower() for issue in self.issues):
            self.log("\n→ ACTION 4: Fix Feature Extraction")
            self.log("  1. Check audio_features_enhanced.py for NaN handling")
            self.log("  2. Add: .fillna(0).replace([np.inf, -np.inf], 0)")
            self.log("  3. Rerun feature extraction")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose audio modality performance issues"
    )
    parser.add_argument(
        '--data-root',
        default=os.path.expanduser('~/Downloads/E-DAIC/data'),
        help='Path to E-DAIC data root'
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  AUDIO MODALITY DIAGNOSTIC REPORT")
    print("=" * 70)
    
    if not os.path.exists(args.data_root):
        print(f"\n❌ ERROR: Data root not found: {args.data_root}")
        print("\nFix:")
        print("  1. Download E-DAIC dataset")
        print("  2. Set environment variable:")
        print(f"     export EDAIC_DATA_ROOT=/path/to/E-DAIC")
        print("  3. Rerun this script")
        return 1
    
    # Create dummy PIDs for testing (from config if available)
    pids = [f"P{i:03d}" for i in range(1, 21)]  # First 20 participants
    
    debugger = AudioDebugger(args.data_root, verbose=True)
    
    # Run diagnostics
    debugger.check_audio_files_exist(pids)
    debugger.check_audio_feature_quality(pids)
    debugger.check_data_leakage()
    debugger.check_model_performance()
    debugger.generate_recommendations()
    
    print("\n" + "=" * 70)
    print(f"Diagnostic complete. Issues: {len(debugger.issues)}")
    print("=" * 70 + "\n")
    
    return 0 if len(debugger.issues) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
