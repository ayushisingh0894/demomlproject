import os
import sys

from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)


from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_trainer_path_obj=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Initiated model trainer')
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            logging.info('Created model_report dict')
            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            logging.info('Got the best model name')
            best_model=models[best_model_name]
            
            logging.info('Saving the object')
            save_object(
                file_path=self.model_trainer_config.model_trainer_path_obj,
                obj= best_model
            )
            
            predicted=best_model.predict(X_test)
            r_square=r2_score(y_test,predicted)

            return best_model

    
        
        except Exception as e:
            raise CustomException(e,sys)