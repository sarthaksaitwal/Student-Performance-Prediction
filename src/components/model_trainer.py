import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def evaluate_models(self,X_train,y_train,X_test,y_test,models):
        report={}

        for model_name,model in models.items():
            model.fit(X_train,y_train)
            y_test_pred=model.predict(X_test)
            r2=r2_score(y_test,y_test_pred)
            report[model_name]=r2
        
        return report

    
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Model training started")

            X_train=train_array[:,:-1]
            y_trian=train_array[:,-1]

            X_test=test_array[:,:-1]
            y_test=test_array[:,-1]

            models={
                "RandomForest":RandomForestRegressor(),
                "LinerRegression":LinearRegression(),
                "GradientBoosting":GradientBoostingRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "XGBoostRegressor":XGBRegressor()
            }

            model_report=self.evaluate_models(
                X_train,y_trian,X_test,y_test,models
            )

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e,sys)
