
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr
import time
import warnings
from typing import Dict, Tuple

class DimensionalityReductionEvaluator:
    """
    精炼版降维质量评估器
    
    专注于三个核心指标：
    - distance_correlation: 距离相关性 (全局结构保留)
    - Q_global: 全局质量指标
    - Q_local: 局部质量指标
    
    特点：
    - 高效的矢量化计算
    - 专注于最重要的评估指标
    - 与单细胞评估框架互补
    """
    
    def __init__(self, verbose=True):
        """
        初始化评估器
        
        参数:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        
    def _log(self, message):
        if self.verbose:
            print(message)
    
    def _validate_inputs(self, X_high, X_low, k):
        """验证输入参数"""
        if not isinstance(X_high, np.ndarray) or not isinstance(X_low, np.ndarray):
            raise TypeError("输入数据必须是numpy数组")
        
        if X_high.shape[0] != X_low.shape[0]:
            raise ValueError(f"高维和低维数据样本数必须相同: {X_high.shape[0]} vs {X_low.shape[0]}")
        
        if k >= X_high.shape[0]:
            raise ValueError(f"k值({k})不能大于或等于样本数({X_high.shape[0]})")
            
        if X_high.ndim != 2 or X_low.ndim != 2:
            raise ValueError("输入数据必须是二维数组")
    
    # ==================== 1. 距离相关性计算 ====================
    
    def distance_correlation_score(self, X_high, X_low):
        """
        计算距离相关性 (Spearman相关)
        评估高维和低维空间中距离的单调关系
        
        参数:
            X_high: 高维空间数据
            X_low: 低维空间数据
            
        返回:
            float: 距离相关性分数 (接近1表示全局结构保留良好)
        """
        try:
            self._log("计算距离矩阵...")
            
            # 计算距离矩阵
            D_high = pairwise_distances(X_high)
            D_low = pairwise_distances(X_low)
            
            # 计算Spearman相关性
            distance_corr, _ = spearmanr(D_high.flatten(), D_low.flatten())
            
            return distance_corr if not np.isnan(distance_corr) else 0.0
            
        except Exception as e:
            warnings.warn(f"距离相关性计算出错: {e}")
            return 0.0
    
    # ==================== 2. 排名矩阵计算 ====================
    
    def get_ranking_matrix(self, distance_matrix):
        """
        计算排名矩阵（优化版本）
        
        参数:
            distance_matrix: 距离矩阵
            
        返回:
            ranking_matrix: 排名矩阵
        """
        try:
            n = len(distance_matrix)
            
            # 使用argsort直接获得排名，避免循环
            sorted_indices = np.argsort(distance_matrix, axis=1)
            
            # 创建排名矩阵
            ranking_matrix = np.zeros((n, n), dtype=np.int32)
            
            # 矢量化操作：为每一行分配排名
            for i in range(n):
                ranking_matrix[i, sorted_indices[i]] = np.arange(n)
            
            # 排除自身（将对角线设为0，其他排名减1）
            mask = np.eye(n, dtype=bool)
            ranking_matrix[~mask] = ranking_matrix[~mask] - 1
            ranking_matrix[mask] = 0
            
            return ranking_matrix
            
        except Exception as e:
            warnings.warn(f"排名矩阵计算出错: {e}")
            return np.zeros((len(distance_matrix), len(distance_matrix)), dtype=np.int32)
    
    # ==================== 3. 共排名矩阵计算 ====================
    
    def get_coranking_matrix(self, rank_high, rank_low):
        """
        计算共排名矩阵（优化版本）
        
        参数:
            rank_high: 高维空间排名矩阵
            rank_low: 低维空间排名矩阵
            
        返回:
            coranking_matrix: 共排名矩阵
        """
        try:
            n = len(rank_high)
            corank = np.zeros((n-1, n-1), dtype=np.int32)
            
            # 矢量化操作：使用numpy的高级索引
            mask = (rank_high > 0) & (rank_low > 0)
            valid_high = rank_high[mask] - 1  # 转换为0-based索引
            valid_low = rank_low[mask] - 1
            
            # 确保索引在有效范围内
            valid_mask = (valid_high < n-1) & (valid_low < n-1)
            valid_high = valid_high[valid_mask]
            valid_low = valid_low[valid_mask]
            
            # 使用np.add.at进行累加
            np.add.at(corank, (valid_high, valid_low), 1)
            
            return corank
            
        except Exception as e:
            warnings.warn(f"共排名矩阵计算出错: {e}")
            n = len(rank_high)
            return np.zeros((n-1, n-1), dtype=np.int32)
    
    # ==================== 4. Q指标计算 ====================
    
    def compute_qnx_series(self, corank):
        """
        计算Q_NX序列
        
        参数:
            corank: 共排名矩阵
            
        返回:
            np.ndarray: Q_NX值序列
        """
        try:
            n = corank.shape[0] + 1
            qnx_values = []
            
            Qnx_cum = 0
            
            for K in range(1, n-1):
                # 计算增量
                if K-1 < corank.shape[0]:
                    intrusions = np.sum(corank[:K, K-1]) if K-1 < corank.shape[1] else 0
                    extrusions = np.sum(corank[K-1, :K]) if K-1 < corank.shape[0] else 0
                    diagonal = corank[K-1, K-1] if K-1 < min(corank.shape) else 0
                    
                    Qnx_increment = intrusions + extrusions - diagonal
                    Qnx_cum += Qnx_increment
                    
                    # 归一化
                    qnx_normalized = Qnx_cum / (K * n)
                    qnx_values.append(qnx_normalized)
            
            return np.array(qnx_values)
            
        except Exception as e:
            warnings.warn(f"Q_NX序列计算出错: {e}")
            return np.array([0.0])
    
    def get_q_local_global(self, qnx_values):
        """
        计算局部和全局质量标量
        
        参数:
            qnx_values: Q_NX值序列
            
        返回:
            tuple: (Q_local, Q_global, K_max)
        """
        try:
            if len(qnx_values) == 0:
                return 0.0, 0.0, 1
            
            # 计算LCMC (Local Continuity Meta-Criterion)
            lcmc = np.copy(qnx_values)
            N = len(qnx_values)
            
            for j in range(N):
                lcmc[j] = lcmc[j] - j/N
            
            K_max = np.argmax(lcmc) + 1
            
            # 计算Q_local和Q_global
            if K_max > 0:
                Q_local = np.mean(qnx_values[:K_max])
            else:
                Q_local = qnx_values[0] if len(qnx_values) > 0 else 0.0
                
            if K_max < len(qnx_values):
                Q_global = np.mean(qnx_values[K_max:])
            else:
                Q_global = qnx_values[-1] if len(qnx_values) > 0 else 0.0
            
            return Q_local, Q_global, K_max
            
        except Exception as e:
            warnings.warn(f"Q指标计算出错: {e}")
            return 0.0, 0.0, 1
    
    # ==================== 5. 综合评估框架 ====================
    
    def comprehensive_evaluation(self, X_high, X_low, k=10):
        """
        综合降维质量评估
        
        参数:
            X_high: 高维空间数据, shape=(n_samples, n_features_high)
            X_low: 低维空间数据, shape=(n_samples, n_features_low)
            k: 考虑的近邻数量
            
        返回:
            dict: 包含核心评估指标的字典
        """        
        # 输入验证
        self._validate_inputs(X_high, X_low, k)
        
        self._log(f"开始降维质量评估 (样本数: {X_high.shape[0]}, k={k})...")
        
        results = {}
        
        # 1. 距离相关性
        self._log("计算距离相关性...")
        results['distance_correlation'] = self.distance_correlation_score(X_high, X_low)
        
        # 2. 计算排名矩阵
        self._log("计算排名矩阵...")
        D_high = pairwise_distances(X_high)
        D_low = pairwise_distances(X_low)
        
        rank_high = self.get_ranking_matrix(D_high)
        rank_low = self.get_ranking_matrix(D_low)
        
        # 3. 计算共排名矩阵
        self._log("计算共排名矩阵...")
        corank = self.get_coranking_matrix(rank_high, rank_low)
        
        # 4. 计算Q指标
        self._log("计算Q指标...")
        qnx_values = self.compute_qnx_series(corank)
        Q_local, Q_global, K_max = self.get_q_local_global(qnx_values)
        
        results['Q_local'] = Q_local
        results['Q_global'] = Q_global
        results['K_max'] = K_max
        
        # 质量评估
        overall_quality = np.mean([
            results['distance_correlation'],
            results['Q_local'],
            results['Q_global']
        ])
        results['overall_quality'] = overall_quality
        
        if self.verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """打印评估结果"""
        
        print("\n" + "="*60)
        print("              降维质量评估结果")
        print("="*60)
        
        print(f"\n【核心质量指标】")
        print(f"  距离相关性: {results['distance_correlation']:.4f} ★")
        print(f"    └─ 接近1表示全局结构保留良好")
        
        print(f"\n  局部质量(Q_local): {results['Q_local']:.4f} ★")
        print(f"    └─ 接近1表示局部结构保留良好")
        
        print(f"\n  全局质量(Q_global): {results['Q_global']:.4f} ★")
        print(f"    └─ 接近1表示全局结构保留良好")
        
        print(f"\n【辅助信息】")
        print(f"  局部-全局分界点(K_max): {results['K_max']}")
        
        # 质量评估
        overall_quality = results['overall_quality']
        
        print(f"\n【综合评估】")
        print(f"  平均质量分数: {overall_quality:.4f}")
        
        if overall_quality >= 0.8:
            quality_level = "优秀"
        elif overall_quality >= 0.6:
            quality_level = "良好"
        elif overall_quality >= 0.4:
            quality_level = "中等"
        else:
            quality_level = "需要改进"
            
        print(f"  质量等级: {quality_level}")
        
        print("="*60)
    
    def compare_methods(self, method_results_dict, k=10):
        """
        比较不同降维方法的效果
        
        参数:
            method_results_dict: {method_name: (X_high, X_low)} 字典
            k: 考虑的近邻数量
            
        返回:
            DataFrame: 比较结果表格
        """
        
        comparison_results = []
        
        for method_name, (X_high, X_low) in method_results_dict.items():
            self._log(f"\n评估方法: {method_name}")
            
            # 暂时关闭详细输出
            original_verbose = self.verbose
            self.verbose = False
            
            results = self.comprehensive_evaluation(X_high, X_low, k)
            
            # 恢复输出设置
            self.verbose = original_verbose
            
            # 计算综合分数
            overall_quality = np.mean([
                results['distance_correlation'],
                results['Q_local'],
                results['Q_global']
            ])
            
            # 添加到比较结果
            comparison_results.append({
                'Method': method_name,
                'Distance_Correlation': results['distance_correlation'],
                'Q_Local': results['Q_local'],
                'Q_Global': results['Q_global'],
                'Overall_Quality': overall_quality,
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(comparison_results)
        
        # 按综合质量排序
        df = df.sort_values('Overall_Quality', ascending=False)
        
        if self.verbose:
            self._print_comparison_table(df)
        
        return df
    
    def _print_comparison_table(self, df):
        """打印比较结果表格"""
        
        print(f"\n{'='*90}")
        print(f"                          降维方法效果比较")
        print('='*90)
        
        # 设置显示格式
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(df.to_string(index=False))
        
        print(f"\n🏆 最佳方法: {df.iloc[0]['Method']} (综合得分: {df.iloc[0]['Overall_Quality']:.4f})")
        
        print('='*90)

# ==================== 便捷函数 ====================

def evaluate_dimensionality_reduction(X_high, X_low, k=10, verbose=True):
    """
    便捷函数：评估降维质量
    
    参数:
        X_high: 高维空间数据
        X_low: 低维空间数据
        k: 考虑的近邻数量
        verbose: 是否详细输出
        
    返回:
        dict: 评估结果
    """
    
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.comprehensive_evaluation(X_high, X_low, k)

def compare_dimensionality_reduction_methods(method_results_dict, k=10, verbose=True):
    """
    便捷函数：比较不同降维方法
    
    参数:
        method_results_dict: {method_name: (X_high, X_low)} 字典
        k: 考虑的近邻数量
        verbose: 是否详细输出
        
    返回:
        DataFrame: 比较结果
    """
    
    evaluator = DimensionalityReductionEvaluator(verbose=verbose)
    return evaluator.compare_methods(method_results_dict, k)