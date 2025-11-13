
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.stats import entropy, skew, kurtosis
from scipy.linalg import svd, norm
import warnings

class SingleCellLatentSpaceEvaluator:
    """
    专门针对单细胞数据的潜在空间质量评估器
    
    特别适用于：
    - 单细胞轨迹数据 (发育、分化等)
    - 单细胞稳态群体数据
    - 时间序列单细胞数据
    
    关键特性：
    - 为轨迹数据调整了指标解释
    - 低各向同性 = 好 (强方向性)
    - 低参与比 = 好 (信息集中)
    - 高谱衰减 = 好 (维度效率)
    """
    
    def __init__(self, data_type="trajectory", verbose=True):
        """
        初始化评估器
        
        参数:
            data_type: "trajectory" 或 "steady_state"
            verbose: 是否输出详细信息
        """
        self.data_type = data_type
        self.verbose = verbose
        
        # 根据数据类型调整期望值
        if data_type == "trajectory":
            self.isotropy_preference = "low"      # 轨迹期望低各向同性
            self.participation_preference = "low"  # 轨迹期望低参与比
        else:  # steady_state
            self.isotropy_preference = "high"     # 稳态期望高各向同性  
            self.participation_preference = "high" # 稳态期望高参与比
    
    def _log(self, message):
        if self.verbose:
            print(message)
    
    # ==================== 1. 修正的流形维度一致性 ====================
    
    def manifold_dimensionality_score_v2(self, latent_space, 
                                        variance_thresholds=[0.8, 0.9, 0.95],
                                        use_multiple_methods=True):
        """
        修正版流形维度一致性评估
        解决了原版本所有方法得分相同的问题
        
        参数:
            latent_space: 潜在空间坐标
            variance_thresholds: 多个方差阈值
            use_multiple_methods: 是否使用多种方法
            
        返回:
            float: 维度效率分数 (0-1)
        """
        try:
            if latent_space.shape[1] == 1:
                return 1.0
            
            # 中心化数据
            centered_data = latent_space - np.mean(latent_space, axis=0)
            
            # PCA分析
            pca = PCA().fit(centered_data)
            explained_variance_ratio = pca.explained_variance_ratio_
            explained_variance = pca.explained_variance_
            
            dimension_scores = []
            
            # 方法1：多阈值维度效率
            for threshold in variance_thresholds:
                cumsum = np.cumsum(explained_variance_ratio)
                effective_dims = np.where(cumsum >= threshold)[0]
                
                if len(effective_dims) > 0:
                    effective_dim = effective_dims[0] + 1
                    # 修正的效率计算：更少维度达到阈值 = 更好
                    efficiency = 1.0 - (effective_dim - 1) / (latent_space.shape[1] - 1)
                    dimension_scores.append(efficiency)
            
            # 方法2：Kaiser准则维度效率
            normalized_eigenvalues = explained_variance / np.mean(explained_variance)
            kaiser_dim = np.sum(normalized_eigenvalues > 1.0)
            kaiser_efficiency = 1.0 - (kaiser_dim - 1) / (latent_space.shape[1] - 1)
            
            # 方法3：肘部法则
            if len(explained_variance) > 2:
                ratios = explained_variance[:-1] / explained_variance[1:]
                elbow_dim = np.argmax(ratios) + 1
                elbow_efficiency = 1.0 - (elbow_dim - 1) / (latent_space.shape[1] - 1)
            else:
                elbow_efficiency = 1.0
            
            # 方法4：谱衰减率
            if len(explained_variance) > 1:
                # 计算特征值的对数衰减
                log_eigenvals = np.log(explained_variance + 1e-10)
                x = np.arange(len(log_eigenvals))
                
                # 线性拟合斜率（衰减率）
                if len(x) > 1:
                    slope = np.polyfit(x, log_eigenvals, 1)[0]
                    # 斜率越负，衰减越快，维度集中度越好
                    decay_score = 1.0 / (1.0 + np.exp(slope))
                else:
                    decay_score = 0.5
            else:
                decay_score = 0.5
            
            # 综合分数
            if use_multiple_methods:
                all_scores = dimension_scores + [kaiser_efficiency, elbow_efficiency, decay_score]
                final_score = np.mean([s for s in all_scores if s is not None])
            else:
                final_score = np.mean(dimension_scores) if dimension_scores else 0.5
            
            return np.clip(final_score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"流形维度一致性计算出错: {e}")
            return 0.5
    
    # ==================== 2. 高效内在特性指标 ====================
    
    def spectral_decay_rate(self, latent_space):
        """谱衰减率 - 越高表示维度集中度越好"""
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            U, s, Vt = svd(centered_data, full_matrices=False)
            eigenvalues = s**2 / (len(latent_space) - 1)
            
            if len(eigenvalues) < 2:
                return 1.0
            
            # 指数衰减拟合
            log_eigenvals = np.log(eigenvalues + 1e-10)
            x = np.arange(len(log_eigenvals))
            
            slope, _ = np.polyfit(x, log_eigenvals, 1)
            
            # 衰减率越负，说明衰减越快
            normalized_decay = 1.0 / (1.0 + np.exp(slope))
            
            # 第一个特征值的集中度
            concentration = eigenvalues[0] / np.sum(eigenvalues)
            
            # 综合分数
            spectral_score = 0.6 * normalized_decay + 0.4 * concentration
            
            return np.clip(spectral_score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"谱衰减率计算出错: {e}")
            return 0.5
    
    def participation_ratio_score(self, latent_space):
        """
        参与比分数
        对于轨迹数据：越低越好 (信息集中)
        对于稳态数据：越高越好 (均匀分布)
        """
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            cov_matrix = np.cov(centered_data.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            if len(eigenvalues) == 0:
                return 0.0
            
            # 参与比公式
            sum_eigenvals = np.sum(eigenvalues)
            sum_eigenvals_squared = np.sum(eigenvalues**2)
            
            if sum_eigenvals_squared > 0:
                participation_ratio = sum_eigenvals**2 / sum_eigenvals_squared
                max_participation = len(eigenvalues)
                normalized_pr = participation_ratio / max_participation
            else:
                normalized_pr = 0.0
            
            # 根据数据类型调整分数
            if self.participation_preference == "low":
                # 轨迹数据：低参与比更好
                score = 1.0 - normalized_pr
            else:
                # 稳态数据：高参与比更好
                score = normalized_pr
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"参与比计算出错: {e}")
            return 0.5
    

    def isotropy_anisotropy_score(self, latent_space):
        """
        各向同性/异性分数 - 增强版
        
        对于轨迹数据：低各向同性更好 (高方向性)
        对于稳态数据：高各向同性更好 (均匀分布)
        
        增强特性：
        - 使用对数变换增加敏感度，避免饱和问题
        - 集成多种测量方法提高区分度
        - 动态调整敏感度阈值
        """
        try:
            centered_data = latent_space - np.mean(latent_space, axis=0)
            cov_matrix = np.cov(centered_data.T)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-12]
            
            if len(eigenvalues) < 2:
                return 1.0
            
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # 方法1：对数椭圆度 (解决饱和问题)
            log_ellipticity = np.log(eigenvalues[0]) - np.log(eigenvalues[-1] + 1e-12)
            enhanced_ellipticity = np.tanh(log_ellipticity / 4.0)
            
            # 方法2：多级条件数 (考虑所有相邻比率)
            condition_ratios = []
            for i in range(len(eigenvalues)-1):
                ratio = eigenvalues[i] / (eigenvalues[i+1] + 1e-12)
                condition_ratios.append(np.log(ratio))
            
            mean_log_condition = np.mean(condition_ratios)
            enhanced_condition = np.tanh(mean_log_condition / 2.0)
            
            # 方法3：比率方差各向异性 (高敏感度)
            ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-12)
            ratio_variance = np.var(np.log(ratios))
            ratio_anisotropy = np.tanh(ratio_variance)
            
            # 方法4：熵各向异性
            eigenval_probs = eigenvalues / np.sum(eigenvalues)
            eigenval_entropy = -np.sum(eigenval_probs * np.log(eigenval_probs + 1e-12))
            max_entropy = np.log(len(eigenvalues))
            entropy_isotropy = eigenval_entropy / max_entropy if max_entropy > 0 else 0
            entropy_anisotropy = 1.0 - entropy_isotropy
            
            # 方法5：主成分支配度
            primary_dominance = eigenvalues[0] / np.sum(eigenvalues[1:]) if len(eigenvalues) > 1 else 1
            dominance_anisotropy = np.tanh(np.log(primary_dominance + 1) / 2.0)
            
            # 方法6：有效维度反比
            participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            effective_dim_anisotropy = 1.0 - (participation_ratio / len(eigenvalues))
            
            # 加权综合分数
            anisotropy_components = [
                enhanced_ellipticity * 0.25,      # 增强椭圆度
                enhanced_condition * 0.25,        # 改进条件数
                ratio_anisotropy * 0.20,          # 比率方差
                entropy_anisotropy * 0.15,        # 熵方法
                dominance_anisotropy * 0.10,      # 主导性
                effective_dim_anisotropy * 0.05   # 有效维度
            ]
            
            weighted_anisotropy = np.sum(anisotropy_components)
            
            # 根据数据类型调整输出
            if self.isotropy_preference == "low":
                # 轨迹数据：高各向异性更好
                score = weighted_anisotropy
            else:
                # 稳态数据：低各向异性更好（高各向同性）
                score = 1.0 - weighted_anisotropy
            
            return np.clip(score, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"各向同性分析出错: {e}")
            return 0.5


    
    # ==================== 3. 单细胞特异性指标 ====================
    
    def trajectory_directionality_score(self, latent_space):
        """
        轨迹方向性评估
        评估主发育轴的支配程度
        """
        try:
            pca = PCA()
            pca.fit(latent_space)
            explained_var = pca.explained_variance_ratio_
            
            if len(explained_var) >= 2:
                # 主方向支配度
                main_dominance = explained_var[0]
                
                # 相对于其他方向的比率
                other_variance = np.sum(explained_var[1:])
                if other_variance > 1e-10:
                    dominance_ratio = explained_var[0] / other_variance
                    # sigmoid 归一化
                    directionality = dominance_ratio / (1.0 + dominance_ratio)
                else:
                    directionality = 1.0
            else:
                directionality = 1.0
                
            return np.clip(directionality, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"轨迹方向性计算出错: {e}")
            return 0.5
    
    def noise_resilience_score(self, latent_space):
        """
        噪声抵抗性评估
        评估降维结果对技术噪声的过滤能力
        """
        try:
            # 基于特征值的噪声评估
            pca = PCA()
            pca.fit(latent_space)
            explained_variance = pca.explained_variance_
            
            if len(explained_variance) > 1:
                # 计算信噪比
                signal_variance = np.sum(explained_variance[:2])  # 前两个主成分
                noise_variance = np.sum(explained_variance[2:]) if len(explained_variance) > 2 else 0
                
                if noise_variance > 1e-10:
                    snr = signal_variance / noise_variance
                    noise_resilience = min(snr / 10.0, 1.0)  # 归一化
                else:
                    noise_resilience = 1.0  # 完美去噪
            else:
                noise_resilience = 1.0
                
            return np.clip(noise_resilience, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"噪声抵抗性计算出错: {e}")
            return 0.5
    
    # ==================== 4. 综合评估框架 ====================
    
    def comprehensive_evaluation(self, latent_space):
        """
        单细胞数据的综合潜在空间评估
        
        参数:
            latent_space: 潜在空间坐标
            
        返回:
            dict: 完整的评估结果
        """
        
        self._log(f"开始单细胞数据 ({self.data_type}) 综合评估...")
        
        results = {}
        
        # 1. 核心流形指标
        self._log("计算流形维度指标...")
        results['manifold_dimensionality'] = self.manifold_dimensionality_score_v2(latent_space)
        
        # 2. 谱分析指标
        self._log("计算谱分析指标...")
        results['spectral_decay_rate'] = self.spectral_decay_rate(latent_space)
        results['participation_ratio'] = self.participation_ratio_score(latent_space)
        results['anisotropy_score'] = self.isotropy_anisotropy_score(latent_space)
        
        # 3. 单细胞特异性指标
        self._log("计算单细胞特异性指标...")
        results['trajectory_directionality'] = self.trajectory_directionality_score(latent_space)
        
        # 4. 技术质量指标
        self._log("计算技术质量指标...")
        results['noise_resilience'] = self.noise_resilience_score(latent_space)
        
        # 5. 计算综合分数
        self._log("计算综合分数...")
        
        # 核心质量分数 (基础流形特性)
        core_metrics = [
            results['manifold_dimensionality'],
            results['spectral_decay_rate'],
            results['participation_ratio'],
            results['anisotropy_score']
        ]
        results['core_quality'] = np.mean(core_metrics)
        
        # 最终综合分数
        if self.data_type == "trajectory":
            # 轨迹数据：更重视方向性
            final_components = [
                results['core_quality'] * 0.5,          # 核心质量 50%
                results['trajectory_directionality'] * 0.3,  # 轨迹方向性 30%
                results['noise_resilience'] * 0.2       # 噪声抵抗 20%
            ]
        else:
            # 稳态数据：更重视核心质量
            final_components = [
                results['core_quality'] * 0.7,          # 核心质量 70%
                results['noise_resilience'] * 0.3       # 噪声抵抗 30%
            ]
        
        results['overall_quality'] = np.sum(final_components)
        
        # 添加解释性信息
        results['data_type'] = self.data_type
        results['interpretation'] = self._generate_interpretation(results)
        
        if self.verbose:
            self._print_comprehensive_results(results)
        
        return results
    
    def _generate_interpretation(self, results):
        """生成结果解释"""
        
        interpretation = {
            'quality_level': '',
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        overall = results['overall_quality']
        
        # 质量等级
        if overall >= 0.8:
            interpretation['quality_level'] = "优秀"
        elif overall >= 0.6:
            interpretation['quality_level'] = "良好"
        elif overall >= 0.4:
            interpretation['quality_level'] = "中等"
        else:
            interpretation['quality_level'] = "需要改进"
        
        # 分析各项指标
        thresholds = {'high': 0.7, 'medium': 0.5, 'low': 0.3}
        
        # 优势分析
        if results['manifold_dimensionality'] > thresholds['high']:
            interpretation['strengths'].append("维度压缩效率高")
        
        if results['spectral_decay_rate'] > thresholds['high']:
            interpretation['strengths'].append("特征值衰减良好")
            
        if results['anisotropy_score'] > thresholds['high']:
            if self.data_type == "trajectory":
                interpretation['strengths'].append("轨迹方向性强")
            else:
                interpretation['strengths'].append("空间分布均匀")
                
        if results['participation_ratio'] > thresholds['high']:
            if self.data_type == "trajectory":
                interpretation['strengths'].append("信息集中度高")
            else:
                interpretation['strengths'].append("维度利用均衡")
        
        if results['trajectory_directionality'] > thresholds['high']:
            interpretation['strengths'].append("主发育轴明显")
        
        # 劣势分析
        if results['noise_resilience'] < thresholds['medium']:
            interpretation['weaknesses'].append("噪声过滤能力不足")
            
        if results['trajectory_directionality'] < thresholds['medium']:
            interpretation['weaknesses'].append("主发育轴不够明显")
            
        if results['core_quality'] < thresholds['medium']:
            interpretation['weaknesses'].append("基础流形质量较低")
        
        # 建议
        if overall < 0.6:
            interpretation['recommendations'].append("考虑调整降维参数")
            interpretation['recommendations'].append("增加数据预处理步骤")
            
        if results['noise_resilience'] < 0.4:
            interpretation['recommendations'].append("增强噪声过滤")
            
        if self.data_type == "trajectory" and results['trajectory_directionality'] < 0.5:
            interpretation['recommendations'].append("优化轨迹方向性保持")
        
        return interpretation
    
    def _print_comprehensive_results(self, results):
        """打印综合评估结果"""
        
        print("\n" + "="*80)
        print(f"           单细胞数据 ({self.data_type.upper()}) 潜在空间质量评估")
        print("="*80)
        
        # 核心指标
        print(f"\n【核心流形指标】")
        print(f"  流形维度一致性: {results['manifold_dimensionality']:.4f} ★")
        print(f"  谱衰减率: {results['spectral_decay_rate']:.4f} (越高越好)")
        print(f"  参与比分数: {results['participation_ratio']:.4f} ({'低参与比好' if self.participation_preference == 'low' else '高参与比好'})")
        print(f"  各向异性分数: {results['anisotropy_score']:.4f} ({'高异性好' if self.isotropy_preference == 'low' else '低异性好'})")
        
        # 单细胞特异性指标
        print(f"\n【单细胞特异性指标】")
        print(f"  轨迹方向性: {results['trajectory_directionality']:.4f} (越高越好)")
        
        # 技术质量
        print(f"\n【技术质量指标】")
        print(f"  噪声抵抗性: {results['noise_resilience']:.4f} (越高越好)")
        
        # 综合评估
        print(f"\n【综合评估】")
        print(f"  核心质量分数: {results['core_quality']:.4f}")
        print(f"  总体质量分数: {results['overall_quality']:.4f} ★★★")
        
        # 解释
        interp = results['interpretation']
        print(f"\n【评估结果】")
        print(f"  质量等级: {interp['quality_level']}")
        
        if interp['strengths']:
            print(f"  优势: {', '.join(interp['strengths'])}")
        
        if interp['weaknesses']:
            print(f"  劣势: {', '.join(interp['weaknesses'])}")
            
        if interp['recommendations']:
            print(f"  建议: {', '.join(interp['recommendations'])}")
        
        print("="*80)
    
    def compare_methods(self, method_results_dict):
        """
        比较不同降维方法的效果
        
        参数:
            method_results_dict: {method_name: latent_space} 字典
            
        返回:
            DataFrame: 比较结果表格
        """
        
        comparison_results = []
        
        for method_name, latent_space in method_results_dict.items():
            self._log(f"\n评估方法: {method_name}")
            
            # 暂时关闭详细输出
            original_verbose = self.verbose
            self.verbose = False
            
            results = self.comprehensive_evaluation(latent_space)
            
            # 恢复输出设置
            self.verbose = original_verbose
            
            # 提取关键指标
            comparison_results.append({
                'Method': method_name,
                'Overall_Quality': results['overall_quality'],
                'Manifold_Dimensionality': results['manifold_dimensionality'],
                'Spectral_Decay': results['spectral_decay_rate'],
                'Participation_Ratio': results['participation_ratio'],
                'Anisotropy_Score': results['anisotropy_score'],
                'Trajectory_Directionality': results['trajectory_directionality'],
                'Noise_Resilience': results['noise_resilience'],
                'Quality_Level': results['interpretation']['quality_level']
            })
        
        # 转换为DataFrame
        df = pd.DataFrame(comparison_results)
        
        # 按总体质量排序
        df = df.sort_values('Overall_Quality', ascending=False)
        
        if self.verbose:
            self._print_comparison_table(df)
        
        return df
    
    def _print_comparison_table(self, df):
        """打印比较结果表格"""
        
        print(f"\n{'='*100}")
        print(f"                    降维方法效果比较 ({self.data_type.upper()} 数据)")
        print('='*100)
        
        # 设置显示格式
        pd.set_option('display.float_format', '{:.4f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        
        print(df.to_string(index=False))
        
        print(f"\n🏆 最佳方法: {df.iloc[0]['Method']} (总分: {df.iloc[0]['Overall_Quality']:.4f})")
        
        print('='*100)

# ==================== 便捷函数 ====================

def evaluate_single_cell_latent_space(latent_space, data_type="trajectory", verbose=True):
    """
    便捷函数：评估单细胞潜在空间质量
    
    参数:
        latent_space: 潜在空间坐标
        data_type: "trajectory" 或 "steady_state"  
        verbose: 是否详细输出
        
    返回:
        dict: 评估结果
    """
    
    evaluator = SingleCellLatentSpaceEvaluator(data_type=data_type, verbose=verbose)
    return evaluator.comprehensive_evaluation(latent_space)

def compare_single_cell_methods(method_results_dict, data_type="trajectory", verbose=True):
    """
    便捷函数：比较不同单细胞降维方法
    
    参数:
        method_results_dict: {method_name: latent_space} 字典
        data_type: "trajectory" 或 "steady_state"
        verbose: 是否详细输出
        
    返回:
        DataFrame: 比较结果
    """
    
    evaluator = SingleCellLatentSpaceEvaluator(data_type=data_type, verbose=verbose)
    return evaluator.compare_methods(method_results_dict)
