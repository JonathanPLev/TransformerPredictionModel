import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from utils.shared_utils import PlayerUtils

class MultiPlayerPredictor:
    """Predictor that works with multi-player dataset and can predict for any player by name"""
    
    def __init__(self, dataset_path="../data_generation/NBA_Multi_Player_Training_Data.csv", mapping_path="../data_generation/player_id_mapping.csv"):
        """
        Initialize predictor with multi-player dataset
        
        Args:
            dataset_path (str): Path to multi-player training dataset
            mapping_path (str): Path to player name -> ID mapping file
        """
        self.dataset_path = dataset_path
        self.mapping_path = mapping_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.data = None
        self.player_mapping = {}
        self.feature_columns = None
        self.is_trained = False
        
        # Load data and mapping
        self.load_data()
        self.load_player_mapping()
        
        # Try to load existing model first
        self.try_load_existing_model()
    
    def load_data(self):
        """Load the multi-player training dataset"""
        try:
            if os.path.exists(self.dataset_path):
                self.data = pd.read_csv(self.dataset_path)
                print(f"Loaded dataset with {len(self.data)} games from {self.data['Player_Name'].nunique()} players")
            else:
                print(f"Dataset not found at {self.dataset_path}")
                print("Please run the MultiPlayerPipeline first to create the dataset")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
    
    def load_player_mapping(self):
        """Load player name to ID mapping"""
        try:
            if os.path.exists(self.mapping_path):
                mapping_df = pd.read_csv(self.mapping_path)
                self.player_mapping = dict(zip(mapping_df['Player_Name'], mapping_df['Player_ID']))
                print(f"Loaded mapping for {len(self.player_mapping)} players")
            else:
                print(f"Player mapping not found at {self.mapping_path}")
        except Exception as e:
            print(f"Error loading player mapping: {str(e)}")
    
    def try_load_existing_model(self):
        """Try to load an existing trained model"""
        
        model_path = "models/multi_player_model.joblib"
        if os.path.exists(model_path):
            try:
                self.load_model(model_path)
                print("✅ Loaded existing trained model")
            except Exception as e:
                print(f"Failed to load model {model_path}: {str(e)}")
        
        print("No existing model found - will train when needed")
    
    def get_player_id_from_name(self, player_name):
        """Get player ID from player name"""
        # First check our mapping
        if player_name in self.player_mapping:
            return self.player_mapping[player_name]
        
        # If not found, try NBA API search
        try:
            return PlayerUtils.get_player_id(player_name)
        except Exception:
            # Try fuzzy matching with existing players
            available_players = list(self.player_mapping.keys())
            matches = [p for p in available_players if player_name.lower() in p.lower()]
            
            if matches:
                print(f"Did you mean one of these players? {matches[:5]}")
                return None
            else:
                print(f"Player '{player_name}' not found in dataset or NBA API")
                return None
    
    def prepare_features(self):
        """Prepare features for training"""
        if self.data is None:
            print("No data loaded")
            return None, None
        
        # Drop current game stats to prevent data leakage
        current_game_stats = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE"
        ]
        
        # Prepare features
        X = self.data.drop(["Beats_Projected_Line", "GAME_DATE", "Player_Name"] + current_game_stats, axis=1, errors='ignore')
        
        # Select only numeric columns
        X = X.select_dtypes(include=[np.number])
        y = self.data["Beats_Projected_Line"]
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        print(f"Prepared features: {len(self.feature_columns)} columns, {len(X)} samples")
        return X, y
    
    def train_model(self):
        """Train the multi-player model"""
        X, y = self.prepare_features()
        
        if X is None:
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        print("Training multi-player model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Players in dataset: {self.data['Player_Name'].nunique()}")
        
        self.is_trained = True
        return accuracy
    
    def predict_for_player(self, player_name, opponent_team=None, is_home_game=True, projected_line=None):
        """
        Make prediction for a specific player
        
        Args:
            player_name (str): Name of the player
            opponent_team (str): Opponent team abbreviation (e.g., 'LAL')
            is_home_game (bool): Whether it's a home game
            projected_line (float): Projected points line for the game
        """
        if not self.is_trained:
            print("Model not trained yet. Please call train_model() first.")
            return None
        
        # Get player ID
        player_id = self.get_player_id_from_name(player_name)
        if player_id is None:
            return None
        
        # Get player's recent data to create features
        try:
            recent_data = self.get_recent_player_data(player_id, player_name)
            if recent_data is None:
                return None
            
            # Create prediction features
            prediction_features = self.create_prediction_features(
                recent_data, opponent_team, is_home_game, projected_line
            )
            
            if prediction_features is None:
                return None
            
            # Make prediction
            prediction_prob = self.model.predict_proba([prediction_features])[0]
            prediction = self.model.predict([prediction_features])[0]
            
            result = {
                'player_name': player_name,
                'player_id': player_id,
                'prediction': 'OVER' if prediction == 1 else 'UNDER',
                'confidence': max(prediction_prob),
                'over_probability': prediction_prob[1],
                'under_probability': prediction_prob[0],
                'projected_line': projected_line,
                'opponent': opponent_team,
                'home_game': is_home_game
            }
            
            return result
            
        except Exception as e:
            print(f"Error making prediction for {player_name}: {str(e)}")
            return None
    
    def get_recent_player_data(self, player_id, player_name):
        """Get recent data for a player to create prediction features"""
        # First try to get from our dataset
        # Check which column name exists in the dataset
        if 'PLAYER_ID' in self.data.columns:
            player_data = self.data[self.data['PLAYER_ID'] == player_id].copy()
        elif 'Player_ID' in self.data.columns:
            player_data = self.data[self.data['Player_ID'] == player_id].copy()
        else:
            print("No Player_ID or PLAYER_ID column found in dataset")
            return None
        
        if len(player_data) > 0:
            # Use most recent game from dataset
            recent_game = player_data.iloc[-1:].copy()
            print(f"Found {len(player_data)} games for {player_name}")
            return recent_game
        else:
            print(f"No recent data found for {player_name} (ID: {player_id}) in dataset")
            print(f"Available columns: {list(self.data.columns)}")
            # Debug: show some player IDs in the dataset
            if 'PLAYER_ID' in self.data.columns:
                unique_ids = self.data['PLAYER_ID'].unique()[:5]
            elif 'Player_ID' in self.data.columns:
                unique_ids = self.data['Player_ID'].unique()[:5]
            else:
                unique_ids = []
            print(f"Sample Player IDs in dataset: {unique_ids}")
            return None
    
    def create_prediction_features(self, recent_data, opponent_team, is_home_game, projected_line):
        """Create features for prediction based on recent player data"""
        try:
            # Start with the most recent game features
            features = recent_data.iloc[0].copy()
            
            # Update game-specific features
            if opponent_team:
                # Reset all team columns to 0
                team_columns = [col for col in self.feature_columns if len(col) == 3 and col.isupper()]
                for col in team_columns:
                    if col in features:
                        features[col] = 0
                
                # Set opponent team to 1
                if opponent_team in features:
                    features[opponent_team] = 1
            
            # Update home/away
            if 'isHomeGame' in features:
                features['isHomeGame'] = 1 if is_home_game else 0
            
            # Update projected line
            if projected_line and 'Projected_Line' in features:
                features['Projected_Line'] = projected_line
            
            # Extract only the features we need for prediction
            prediction_features = []
            for col in self.feature_columns:
                if col in features:
                    prediction_features.append(features[col])
                else:
                    prediction_features.append(0)  # Default value for missing features
            
            return prediction_features
            
        except Exception as e:
            print(f"Error creating prediction features: {str(e)}")
            return None
    
    def save_model(self, model_path="models/multi_player_model.joblib"):
        """Save the trained model"""
        if not self.is_trained:
            print("No trained model to save")
            return
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'player_mapping': self.player_mapping
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path="models/multi_player_model.joblib"):
        """Load a saved model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.player_mapping = model_data['player_mapping']
            self.is_trained = True
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def list_available_players(self):
        """List all players available in the dataset"""
        if self.data is not None:
            players = sorted(self.data['Player_Name'].unique())
            print(f"Available players ({len(players)}):")
            for i, player in enumerate(players, 1):
                print(f"{i:2d}. {player}")
            return players
        else:
            print("No data loaded")
            return []


if __name__ == "__main__":
    # Example usage
    predictor = MultiPlayerPredictor()
    
    if predictor.data is not None:
        # Train the model only if not already trained
        if not predictor.is_trained:
            print("Training new model...")
            accuracy = predictor.train_model()
            if accuracy:
                # Save the model
                predictor.save_model()
        else:
            print("Using existing trained model")
            
        # List available players
        predictor.list_available_players()
        
        # Example predictions
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        result = predictor.predict_for_player(
            player_name="",
            opponent_team="LAL",
            is_home_game=True,
            projected_line=26.5
        )
        
        if result:
            print(f"\nPrediction for {result['player_name']}:")
            print(f"Opponent: {result['opponent']} ({'Home' if result['home_game'] else 'Away'})")
            print(f"Projected Line: {result['projected_line']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Over Probability: {result['over_probability']:.3f}")
            print(f"Under Probability: {result['under_probability']:.3f}")
