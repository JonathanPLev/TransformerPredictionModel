#!/usr/bin/env python3
"""
Command-line interface for making NBA player predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_player_predictor import MultiPlayerPredictor
import argparse

def main():
    parser = argparse.ArgumentParser(description='Predict NBA player performance')
    parser.add_argument('player_name', help='Player name (e.g., "Franz Wagner")')
    parser.add_argument('--opponent', '-o', help='Opponent team abbreviation (e.g., LAL)')
    parser.add_argument('--home', action='store_true', help='Home game (default: away)')
    parser.add_argument('--line', '-l', type=float, help='Projected points line')
    parser.add_argument('--list-players', action='store_true', help='List all available players')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("Loading multi-player predictor...")
    predictor = MultiPlayerPredictor()
    
    if predictor.data is None:
        print("Error: No dataset found. Please run the MultiPlayerPipeline first.")
        return
    
    # List players if requested
    if args.list_players:
        predictor.list_available_players()
        return
    
    # Train model if not already trained
    if not predictor.is_trained:
        print("Training model...")
        accuracy = predictor.train_model()
        if accuracy:
            predictor.save_model()
        else:
            print("Failed to train model")
            return
    
    # Make prediction
    print(f"\nMaking prediction for {args.player_name}...")
    
    result = predictor.predict_for_player(
        player_name=args.player_name,
        opponent_team=args.opponent,
        is_home_game=args.home,
        projected_line=args.line
    )
    
    if result:
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Player: {result['player_name']}")
        print(f"Player ID: {result['player_id']}")
        
        if result['opponent']:
            print(f"Opponent: {result['opponent']}")
        
        print(f"Game Type: {'Home' if result['home_game'] else 'Away'}")
        
        if result['projected_line']:
            print(f"Projected Line: {result['projected_line']} points")
        
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Over Probability: {result['over_probability']:.1%}")
        print(f"Under Probability: {result['under_probability']:.1%}")
        
        # Add recommendation
        if result['confidence'] > 0.7:
            confidence_level = "HIGH"
        elif result['confidence'] > 0.6:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        print(f"\nRecommendation: {confidence_level} confidence {result['prediction']}")
    else:
        print("Failed to make prediction")

if __name__ == "__main__":
    main()
