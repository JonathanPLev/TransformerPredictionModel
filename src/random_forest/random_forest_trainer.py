from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from datetime import datetime

class RandomForestTrainer:
    def __init__(self, data, target, shuffle_data=True):
        self.data = pd.read_csv(data)
        if shuffle_data:
            self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        self.target = target
        self.model = RandomForestClassifier()


    def train_model(self):
        # Drop target and date columns
        X = self.data.drop([self.target, "GAME_DATE"], axis=1)
        
        # Drop Franz's current game stats to prevent data leakage
        current_game_stats = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", 
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", 
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE",  

        ]

        cols_to_drop = [col for col in current_game_stats if col in X.columns]
        if cols_to_drop:
            print(f"Dropping current game stats: {cols_to_drop}")
        X = X.drop(cols_to_drop, axis=1)

        X = X.select_dtypes(include=[np.number])
        y = self.data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Dataset size: {len(self.data)}")
        print(f"Test size: {len(X_test)}")
        print(f"Class distribution:\n{y.value_counts()}")
        print(f"Features used: {len(X.columns)}")

    def cross_validate_model(self, cv_folds=5, scoring='accuracy'):
        """Perform cross-validation on the model"""
        # Prepare features same as train_model
        X = self.data.drop([self.target, "GAME_DATE"], axis=1)
        
        # Drop Franz's current game stats to prevent data leakage
        current_game_stats = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE",
        ]
        
        cols_to_drop = [col for col in current_game_stats if col in X.columns]
        if cols_to_drop:
            print(f"Dropping current game stats for CV: {cols_to_drop}")
        X = X.drop(cols_to_drop, axis=1)
        
        X = X.select_dtypes(include=[np.number])
        y = self.data[self.target]
        
        # Perform cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=kfold, scoring=scoring)
        print(f"\n=== {cv_folds}-Fold Cross-Validation Results ===")
        print(f"Individual fold scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.4f}")
        print(f"Standard Deviation: {cv_scores.std():.4f}")
        print(f"95% Confidence Interval: [{cv_scores.mean() - 2*cv_scores.std():.4f}, {cv_scores.mean() + 2*cv_scores.std():.4f}]")
        
        return cv_scores

    def cross_validate_with_models(self, cv_folds=5, return_best=True, create_ensemble=False):
        """Cross-validation that returns the actual trained models"""
        # Prepare features
        X = self.data.drop([self.target, "GAME_DATE"], axis=1)
        
        current_game_stats = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE"
        ]
        
        cols_to_drop = [col for col in current_game_stats if col in X.columns]
        X = X.drop(cols_to_drop, axis=1)
        X = X.select_dtypes(include=[np.number])
        y = self.data[self.target]
        
        # Manual cross-validation to get the trained models
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        fold_models = []
        fold_scores = []
        
        print(f"\n=== Training {cv_folds} models via Cross-Validation ===")
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            # Split data for this fold
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = RandomForestClassifier(random_state=42)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # Evaluate on validation set
            y_pred = fold_model.predict(X_val_fold)
            fold_accuracy = accuracy_score(y_val_fold, y_pred)
            
            fold_models.append(fold_model)
            fold_scores.append(fold_accuracy)
            
            print(f"Fold {fold_idx + 1}: Accuracy = {fold_accuracy:.4f}")
        
        print(f"Mean CV Accuracy: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
        
        if create_ensemble:
            estimators = [(f'model_{i}', model) for i, model in enumerate(fold_models)]
            ensemble_model = VotingClassifier(estimators=estimators, voting='soft')

            ensemble_model.fit(X, y)
            self.model = ensemble_model
            print("Created ensemble model from all CV folds")
            
            return fold_scores, fold_models, ensemble_model
        
        elif return_best:
            best_idx = np.argmax(fold_scores)
            best_model = fold_models[best_idx]
            best_score = fold_scores[best_idx]
            
            best_model.fit(X, y)
            self.model = best_model
            
            print(f"Selected best model from fold {best_idx + 1} (accuracy: {best_score:.4f})")
            return fold_scores, fold_models, best_model
        
        else:
            return fold_scores, fold_models

    def load_model(self, model_path, method='auto'):
        """Load a saved model from disk"""
        if method == 'auto':
            # Auto-detect based on file extension
            if model_path.endswith('.joblib'):
                method = 'joblib'
            elif model_path.endswith('.pkl'):
                method = 'pickle'
            else:
                raise ValueError("Cannot auto-detect method. Specify 'joblib' or 'pickle'")
        
        if method == 'joblib':
            self.model = joblib.load(model_path)
            print(f"Model loaded using joblib: {model_path}")
        
        elif method == 'pickle':
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded using pickle: {model_path}")
        
        else:
            raise ValueError("Method must be 'joblib' or 'pickle'")

    def save_model_with_metadata(self, model_name="franz_wagner_rf", accuracy=None):
        """Save model with metadata for easy tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/{model_name}_{timestamp}.joblib"
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_path': model_path,
            'timestamp': timestamp,
            'accuracy': accuracy,
            'target': self.target,
            'dataset_size': len(self.data)
        }
        
        # Add model-specific info
        if hasattr(self.model, 'estimators_'):
            # VotingClassifier ensemble
            metadata['model_type'] = 'VotingClassifier'
            metadata['n_estimators'] = len(self.model.estimators_)
            metadata['voting'] = getattr(self.model, 'voting', 'unknown')
        elif hasattr(self.model, 'n_estimators'):
            # Single RandomForestClassifier
            metadata['model_type'] = 'RandomForestClassifier'
            metadata['n_estimators'] = self.model.n_estimators
        else:
            metadata['model_type'] = str(type(self.model).__name__)
        
        metadata_path = f"models/{model_name}_{timestamp}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Model saved: {model_path}")
        print(f"Metadata saved: {metadata_path}")
        return model_path, metadata_path


if __name__ == "__main__":
    trainer = RandomForestTrainer("../data_generation/Franz_Wagner_training_data.csv", "Beats_Projected_Line", shuffle_data=True)
    
    # # Option 1: Get the best model from cross-validation
    cv_scores, all_models, best_model = trainer.cross_validate_with_models(return_best=True)
    mean_accuracy = np.mean(cv_scores)
    trainer.save_model_with_metadata("franz_wagner_best", accuracy=mean_accuracy)

    # Option 2: Create ensemble from all CV models
    # cv_scores, all_models, ensemble = trainer.cross_validate_with_models(create_ensemble=True)
    # trainer.save_model_with_metadata("franz_wagner_ensemble", accuracy=np.mean(cv_scores))
