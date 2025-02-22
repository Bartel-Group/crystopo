from typing import Dict, Tuple, List, Optional
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import joblib

class BettiClassifier:
    """Classifier for structural prototypes based on Betti curves."""
    
    def __init__(self, 
                 n_estimators: int = 100,
                 test_size: float = 0.2,
                 random_state: Optional[int] = None) -> None:
        """
        Initialize the classifier.
        
        Args:
            n_estimators: Number of trees in random forest
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.test_size = test_size
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
    def _prepare_data(self,
                     data: Dict[str, List[Tuple[NDArray[np.float64], 
                                              NDArray[np.float64], 
                                              NDArray[np.float64]]]]
                     ) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Convert dictionary of Betti curves into format for training.
        
        Args:
            data: Dictionary mapping prototype labels to lists of Betti curve tuples
                 Each tuple contains (betti0, betti1, betti2)
            
        Returns:
            X: Array of concatenated Betti curves
            y: Array of numeric labels
        """
        # Create label mappings if they don't exist
        if not self.label_to_idx:
            self.label_to_idx = {label: i for i, label in enumerate(data.keys())}
            self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        
        X_list = []
        y_list = []
        
        # For each prototype label
        for label, curves_list in data.items():
            label_idx = self.label_to_idx[label]
            
            # For each set of Betti curves for this prototype
            for curves in curves_list:
                betti0, betti1, betti2 = curves
                combined = np.concatenate([betti0, betti1, betti2])
                
                X_list.append(combined)
                y_list.append(label_idx)
        
        return np.array(X_list), np.array(y_list)
    
    def train(self,
             data: Dict[str, List[Tuple[NDArray[np.float64], 
                                      NDArray[np.float64], 
                                      NDArray[np.float64]]]]
             ) -> Tuple[float, float]:
        """
        Train the classifier and evaluate on test set.
        
        Args:
            data: Dictionary mapping prototype labels to lists of Betti curve tuples
                 Each tuple contains (betti0, betti1, betti2)
            
        Returns:
            train_accuracy: Accuracy on training set
            test_accuracy: Accuracy on test set
        """
        # Print dataset statistics
        print("Dataset statistics:")
        for label, curves_list in data.items():
            print(f"{label}: {len(curves_list)} examples")
            
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Compute accuracies
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        return train_accuracy, test_accuracy
    
    def save(self, filepath: str) -> None:
        """
        Save the trained classifier to a file.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ValueError: If model hasn't been trained yet
        """
        if not hasattr(self.model, 'classes_'):
            raise ValueError("Model must be trained before saving")
        
        # Create save path if it doesn't exist
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save everything needed to restore the classifier
        save_dict = {
            'model': self.model,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label
        }
        
        joblib.dump(save_dict, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BettiClassifier':
        """
        Load a trained classifier from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BettiClassifier instance
            
        Raises:
            FileNotFoundError: If save file doesn't exist
        """
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"No saved model found at {load_path}")
        
        # Load the saved dictionary
        save_dict = joblib.load(load_path)
        
        # Create new instance
        instance = cls()
        instance.model = save_dict['model']
        instance.label_to_idx = save_dict['label_to_idx']
        instance.idx_to_label = save_dict['idx_to_label']
        
        print(f"Model loaded from {load_path}")
        return instance
    
    def predict(self,
               curves: Tuple[NDArray[np.float64], 
                           NDArray[np.float64], 
                           NDArray[np.float64]]
               ) -> str:
        """
        Predict the prototype label for a new set of Betti curves.
        
        Args:
            curves: Tuple of (betti0, betti1, betti2) curves
            
        Returns:
            Predicted prototype label
        """
        if not self.label_to_idx:
            raise ValueError("Model must be trained before making predictions")
            
        betti0, betti1, betti2 = curves
        X = np.concatenate([betti0, betti1, betti2]).reshape(1, -1)
        y_pred = self.model.predict(X)[0]
        return self.idx_to_label[y_pred]
    
    def predict_proba(self,
                     curves: Tuple[NDArray[np.float64], 
                                 NDArray[np.float64], 
                                 NDArray[np.float64]]
                     ) -> Dict[str, float]:
        """
        Get prediction probabilities for each prototype.
        
        Args:
            curves: Tuple of (betti0, betti1, betti2) curves
            
        Returns:
            Dictionary mapping prototype labels to probabilities
        """
        if not self.label_to_idx:
            raise ValueError("Model must be trained before making predictions")
            
        betti0, betti1, betti2 = curves
        X = np.concatenate([betti0, betti1, betti2]).reshape(1, -1)
        probs = self.model.predict_proba(X)[0]
        
        return {self.idx_to_label[i]: prob for i, prob in enumerate(probs)}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance of each Betti curve region.
        
        Returns:
            Dictionary mapping curve names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model must be trained before getting feature importance")
            
        n_features = len(self.model.feature_importances_)
        region_size = n_features // 3
        
        importances = {
            'β₀': np.mean(self.model.feature_importances_[:region_size]),
            'β₁': np.mean(self.model.feature_importances_[region_size:2*region_size]),
            'β₂': np.mean(self.model.feature_importances_[2*region_size:])
        }
        
        return importances

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluates the classifier performance."""
        return self.model.score(X, y)

