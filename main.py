import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    """
    DeFi Credit Scoring System for Aave V2 Protocol
    Generates credit scores (0-1000) based on transaction behavior
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.feature_weights = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and basic preprocessing of DeFi transaction data"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        df = pd.json_normalize(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract core fields
        df['action'] = df['action'].astype(str)
        df['amount'] = pd.to_numeric(df['actionData.amount'], errors='coerce')
        df['asset_price_usd'] = pd.to_numeric(df['actionData.assetPriceUSD'], errors='coerce')
        df['asset_symbol'] = df['actionData.assetSymbol']
        df['wallet'] = df['userWallet']
        df['protocol'] = df['protocol']
        df['network'] = df['network']
        
        # Calculate USD value of transactions
        df['usd_value'] = df['amount'] * df['asset_price_usd']
        
        # Drop rows with critical missing values
        df.dropna(subset=['amount', 'asset_price_usd', 'wallet'], inplace=True)
        
        return df
    
    def engineer_credit_features(self, df):
        """Engineer features specifically designed for credit scoring"""
        
        features_list = []
        
        for wallet in df['wallet'].unique():
            user_data = df[df['wallet'] == wallet].copy()
            user_data = user_data.sort_values('timestamp')
            
            features = {'wallet': wallet}
            
            # === RELIABILITY INDICATORS ===
            features['total_transactions'] = len(user_data)
            features['unique_days_active'] = user_data['timestamp'].dt.date.nunique()
            features['account_age_days'] = (user_data['timestamp'].max() - user_data['timestamp'].min()).days
            features['avg_daily_activity'] = features['total_transactions'] / max(1, features['unique_days_active'])
            
            # === FINANCIAL STABILITY METRICS ===
            features['total_usd_volume'] = user_data['usd_value'].sum()
            features['avg_txn_size'] = user_data['usd_value'].mean()
            features['median_txn_size'] = user_data['usd_value'].median()
            features['txn_size_consistency'] = 1 / (1 + user_data['usd_value'].std() / max(user_data['usd_value'].mean(), 1e-10))
            
            # === RESPONSIBLE USAGE PATTERNS ===
            action_counts = user_data['action'].value_counts()
            total_actions = len(user_data)
            
            features['deposit_ratio'] = action_counts.get('deposit', 0) / total_actions
            features['withdrawal_ratio'] = action_counts.get('redeemunderlying', 0) / total_actions
            features['borrow_ratio'] = action_counts.get('borrow', 0) / total_actions
            features['repay_ratio'] = action_counts.get('repay', 0) / total_actions
            features['liquidation_ratio'] = action_counts.get('liquidationcall', 0) / total_actions
            
            # Healthy financial behavior indicators
            deposits = user_data[user_data['action'] == 'deposit']['usd_value'].sum()
            withdrawals = user_data[user_data['action'] == 'redeemunderlying']['usd_value'].sum()
            borrows = user_data[user_data['action'] == 'borrow']['usd_value'].sum()
            repays = user_data[user_data['action'] == 'repay']['usd_value'].sum()
            
            features['net_deposit_flow'] = deposits - withdrawals
            features['borrow_repay_ratio'] = repays / max(borrows, 1e-10) if borrows > 0 else 1.0
            features['deposit_to_borrow_ratio'] = deposits / max(borrows, 1e-10) if borrows > 0 else 10.0
            
            # === DIVERSIFICATION & SOPHISTICATION ===
            features['asset_diversity'] = user_data['asset_symbol'].nunique()
            features['protocol_diversity'] = user_data['protocol'].nunique()
            
            # Asset concentration (lower is better for credit)
            if len(user_data) > 1:
                asset_volumes = user_data.groupby('asset_symbol')['usd_value'].sum()
                asset_shares = asset_volumes / asset_volumes.sum()
                features['asset_concentration'] = sum(share**2 for share in asset_shares)
            else:
                features['asset_concentration'] = 1.0
            
            # === RISK INDICATORS (NEGATIVE SIGNALS) ===
            # Rapid transaction patterns (potential bot behavior)
            if len(user_data) > 1:
                time_diffs = user_data['timestamp'].diff().dt.total_seconds() / 3600
                features['avg_time_between_txns'] = time_diffs.mean()
                features['rapid_txn_count'] = (time_diffs < 0.1).sum()  # < 6 minutes
                features['rapid_txn_ratio'] = features['rapid_txn_count'] / max(len(time_diffs) - 1, 1)
            else:
                features['avg_time_between_txns'] = 24
                features['rapid_txn_count'] = 0
                features['rapid_txn_ratio'] = 0
            
            # Large transaction spikes (potential manipulation)
            features['max_txn_size'] = user_data['usd_value'].max()
            features['large_txn_ratio'] = (user_data['usd_value'] > user_data['usd_value'].quantile(0.9)).sum() / total_actions
            
            # Liquidation involvement (very negative)
            features['liquidation_count'] = action_counts.get('liquidationcall', 0)
            features['was_liquidated'] = 1 if features['liquidation_count'] > 0 else 0
            
            # === BEHAVIORAL CONSISTENCY ===
            # Regular usage pattern
            if features['account_age_days'] > 0:
                features['activity_consistency'] = features['unique_days_active'] / max(features['account_age_days'], 1)
            else:
                features['activity_consistency'] = 1.0
            
            # Time-based patterns
            user_data['hour'] = user_data['timestamp'].dt.hour
            user_data['day_of_week'] = user_data['timestamp'].dt.dayofweek
            
            # Spread of activity (more spread = more human-like)
            features['hour_entropy'] = self._calculate_entropy(user_data['hour'])
            features['day_entropy'] = self._calculate_entropy(user_data['day_of_week'])
            
            # === ADVANCED BEHAVIORAL SIGNALS ===
            # Transaction value progression (learning/growth pattern)
            if len(user_data) >= 5:
                recent_avg = user_data.tail(5)['usd_value'].mean()
                early_avg = user_data.head(5)['usd_value'].mean()
                features['value_growth'] = (recent_avg - early_avg) / max(early_avg, 1e-10)
            else:
                features['value_growth'] = 0
            
            # Seasonal patterns (more sophisticated users)
            features['unique_months_active'] = user_data['timestamp'].dt.month.nunique()
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_entropy(self, series):
        """Calculate entropy of a series"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    def calculate_credit_scores(self, features_df):
        """Calculate credit scores based on engineered features"""
        
        # Prepare features for ML
        wallet_col = features_df['wallet']
        feature_cols = [col for col in features_df.columns if col != 'wallet']
        
        # Handle infinite and missing values
        features_clean = features_df[feature_cols].copy()
        features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.fillna(features_clean.median())
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        # === SCORING COMPONENTS ===
        
        # 1. Anomaly Detection (0-300 points)
        anomaly_scores = self.isolation_forest.fit_predict(features_scaled)
        anomaly_component = np.where(anomaly_scores == 1, 300, 100)  # Normal=300, Anomaly=100
        
        # 2. Behavioral Clustering (0-250 points)
        cluster_labels = self.kmeans.fit_predict(features_scaled)
        cluster_component = self._score_clusters(features_df, cluster_labels)
        
        # 3. Rule-Based Scoring (0-450 points)
        rule_component = self._rule_based_scoring(features_df)
        
        # Combine components
        total_scores = anomaly_component + cluster_component + rule_component
        
        # Normalize to 0-1000 range
        min_score, max_score = total_scores.min(), total_scores.max()
        normalized_scores = 50 + (total_scores - min_score) / (max_score - min_score) * 900
        
        # Create results dataframe
        results = pd.DataFrame({
            'wallet': wallet_col,
            'credit_score': normalized_scores.round(0).astype(int),
            'anomaly_component': anomaly_component,
            'cluster_component': cluster_component,
            'rule_component': rule_component,
            'cluster_label': cluster_labels
        })
        
        return results
    
    def _score_clusters(self, features_df, cluster_labels):
        """Score users based on cluster characteristics"""
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            # Define "good" cluster characteristics
            avg_volume = cluster_data['total_usd_volume'].mean()
            avg_consistency = cluster_data['txn_size_consistency'].mean()
            avg_repay_ratio = cluster_data['borrow_repay_ratio'].mean()
            avg_rapid_ratio = cluster_data['rapid_txn_ratio'].mean()
            avg_liquidation_ratio = cluster_data['liquidation_ratio'].mean()
            
            # Score cluster (higher is better)
            cluster_score = (
                min(avg_volume / 10000, 1.0) * 100 +  # Volume bonus (capped)
                avg_consistency * 50 +  # Consistency bonus
                min(avg_repay_ratio, 2.0) * 50 +  # Repayment bonus (capped)
                max(0, (1 - avg_rapid_ratio)) * 30 +  # Penalty for rapid txns
                max(0, (1 - avg_liquidation_ratio)) * 20  # Penalty for liquidations
            )
            
            cluster_stats[cluster_id] = min(cluster_score, 250)  # Cap at 250
        
        # Assign scores based on cluster membership
        cluster_scores = np.array([cluster_stats[label] for label in cluster_labels])
        
        return cluster_scores
    
    def _rule_based_scoring(self, features_df):
        """Rule-based scoring component"""
        
        scores = np.zeros(len(features_df))
        
        for i, row in features_df.iterrows():
            score = 0
            
            # === POSITIVE INDICATORS ===
            
            # Account maturity (0-50 points)
            score += min(row['account_age_days'] / 365 * 50, 50)
            
            # Transaction volume (0-50 points)
            score += min(np.log10(max(row['total_usd_volume'], 1)) * 10, 50)
            
            # Activity consistency (0-40 points)
            score += row['activity_consistency'] * 40
            
            # Repayment behavior (0-60 points)
            score += min(row['borrow_repay_ratio'], 1.0) * 60
            
            # Diversification (0-30 points)
            score += min(row['asset_diversity'] / 5 * 30, 30)
            
            # Regular usage (0-40 points)
            score += min(row['total_transactions'] / 100 * 40, 40)
            
            # Behavioral entropy (0-30 points)
            score += (row['hour_entropy'] + row['day_entropy']) / 2 * 30
            
            # === NEGATIVE INDICATORS ===
            
            # Rapid transactions penalty (0 to -50 points)
            score -= row['rapid_txn_ratio'] * 50
            
            # Liquidation penalty (0 to -100 points)
            score -= row['liquidation_ratio'] * 100
            
            # Large transaction spikes penalty (0 to -30 points)
            score -= row['large_txn_ratio'] * 30
            
            # Asset concentration penalty (0 to -20 points)
            score -= (row['asset_concentration'] - 0.5) * 20 if row['asset_concentration'] > 0.5 else 0
            
            # === SEVERE PENALTIES ===
            
            # Heavy liquidation involvement
            if row['liquidation_count'] > 5:
                score -= 100
            
            # Suspicious rapid activity
            if row['rapid_txn_count'] > 10:
                score -= 50
            
            # Ensure score is within bounds
            scores[i] = max(0, min(score, 450))
        
        return scores
    
    def generate_score_explanations(self, results, features_df):
        """Generate explanations for credit scores"""
        
        explanations = []
        
        for i, row in results.iterrows():
            wallet = row['wallet']
            score = row['credit_score']
            user_features = features_df[features_df['wallet'] == wallet].iloc[0]
            
            explanation = {
                'wallet': wallet,
                'credit_score': score,
                'score_tier': self._get_score_tier(score),
                'key_factors': self._get_key_factors(user_features),
                'risk_flags': self._get_risk_flags(user_features),
                'recommendations': self._get_recommendations(user_features)
            }
            
            explanations.append(explanation)
        
        return pd.DataFrame(explanations)
    
    def _get_score_tier(self, score):
        """Get score tier description"""
        if score >= 800:
            return "Excellent (800-1000)"
        elif score >= 700:
            return "Good (700-799)"
        elif score >= 600:
            return "Fair (600-699)"
        elif score >= 500:
            return "Poor (500-599)"
        else:
            return "Very Poor (0-499)"
    
    def _get_key_factors(self, user_features):
        """Identify key positive factors"""
        factors = []
        
        if user_features['borrow_repay_ratio'] > 0.8:
            factors.append("Strong repayment history")
        
        if user_features['account_age_days'] > 180:
            factors.append("Mature account")
        
        if user_features['total_usd_volume'] > 50000:
            factors.append("High transaction volume")
        
        if user_features['asset_diversity'] > 3:
            factors.append("Diversified asset usage")
        
        if user_features['activity_consistency'] > 0.1:
            factors.append("Consistent activity pattern")
        
        return factors[:3]  # Top 3 factors
    
    def _get_risk_flags(self, user_features):
        """Identify risk flags"""
        flags = []
        
        if user_features['liquidation_ratio'] > 0:
            flags.append("Liquidation history")
        
        if user_features['rapid_txn_ratio'] > 0.1:
            flags.append("High-frequency trading pattern")
        
        if user_features['borrow_repay_ratio'] < 0.5:
            flags.append("Poor repayment behavior")
        
        if user_features['large_txn_ratio'] > 0.2:
            flags.append("Irregular transaction sizes")
        
        return flags
    
    def _get_recommendations(self, user_features):
        """Get improvement recommendations"""
        recommendations = []
        
        if user_features['borrow_repay_ratio'] < 0.8:
            recommendations.append("Improve repayment consistency")
        
        if user_features['asset_diversity'] < 2:
            recommendations.append("Diversify asset portfolio")
        
        if user_features['rapid_txn_ratio'] > 0.1:
            recommendations.append("Reduce high-frequency trading")
        
        return recommendations
    
    def run_complete_analysis(self, file_path):
        """Run complete credit scoring analysis"""
        
        print("🔄 Loading and preprocessing data...")
        df = self.load_and_preprocess_data(file_path)
        print(f"✅ Loaded {len(df)} transactions for {df['wallet'].nunique()} wallets")
        
        print("\n🔄 Engineering credit features...")
        features_df = self.engineer_credit_features(df)
        print(f"✅ Created {len(features_df.columns)-1} features")
        
        print("\n🔄 Calculating credit scores...")
        results = self.calculate_credit_scores(features_df)
        print("✅ Credit scores calculated")
        
        print("\n🔄 Generating explanations...")
        explanations = self.generate_score_explanations(results, features_df)
        print("✅ Explanations generated")
        
        # Save results
        output_dir = r"C:\Users\91997\Desktop\defi_credit_scoring_project\data"
        results.to_csv(f"{output_dir}/credit_scores.csv", index=False)
        explanations.to_csv(f"{output_dir}/credit_explanations.csv", index=False)
        features_df.to_csv(f"{output_dir}/credit_features.csv", index=False)
        
        print(f"\n✅ Results saved to {output_dir}/")
        
        # Print summary statistics
        print("\n📊 Credit Score Summary:")
        print(f"Average Score: {results['credit_score'].mean():.0f}")
        print(f"Median Score: {results['credit_score'].median():.0f}")
        print(f"Score Range: {results['credit_score'].min():.0f} - {results['credit_score'].max():.0f}")
        
        print("\n🎯 Score Distribution:")
        score_ranges = [
            (800, 1000, "Excellent"),
            (700, 799, "Good"),
            (600, 699, "Fair"),
            (500, 599, "Poor"),
            (0, 499, "Very Poor")
        ]
        
        for min_score, max_score, tier in score_ranges:
            count = ((results['credit_score'] >= min_score) & (results['credit_score'] <= max_score)).sum()
            pct = count / len(results) * 100
            print(f"{tier}: {count} wallets ({pct:.1f}%)")
        
        return results, explanations, features_df

# Main execution
if __name__ == "__main__":
    # Initialize the credit scorer
    scorer = DeFiCreditScorer()
    
    # Run complete analysis
    file_path = r"C:\Users\91997\Desktop\defi_credit_scoring_project\data\user-wallet-transactions.json"
    results, explanations, features = scorer.run_complete_analysis(file_path)
    
    print("\n🎉 Credit scoring analysis complete!")
    print("\nFiles generated:")
    print("• credit_scores.csv - Final credit scores")
    print("• credit_explanations.csv - Score explanations")
    print("• credit_features.csv - Engineered features")