from DimRed import *


class Evaluation:
    def __init__(
        self,
        _data: Dict[str, np.ndarray],
        all_possible_variations: Dict[str, List],
        labels: np.ndarray,
        metric: str = "accuracy",
        sklearn_config: Dict[Any, Dict[str, Union[str, int]]] = sklearn_config,
        lgb_config: Dict[str, Union[str, int]] = lgb_config,
        xgb_config: Dict[str, Union[str, int]] = xgb_config,
    ) -> None:
        self.sklearn_config = sklearn_config
        self.lgb_config = lgb_config
        self.xgb_config = xgb_config
        self._data = _data
        self.all_variations = all_possible_variations
        self.labels = labels
        self.metric = metric

    def sklearn(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        inner_iterator: tqdm,
        results: Dict = {},
        dimred_technique: str = None,
    ) -> Tuple[Dict[str, Union[str, int]], Dict[str, int]]:
        best_model = [0, {}]
        for model in tqdm(self.sklearn_config):
            print(model)
            name = dimred_technique + model().__class__.__name__
            inner_iterator.set_description(name)
            model_config = self.sklearn_config[model]
            wandb.init(
                project=PROJECT_NAME,
                name=name,
                config={
                    "model": name,
                    "results": results,
                    "modelLibrary": "sklearn",
                    "config": model_config,
                },
            )
            model = RandomizedSearchCV(model(), model_config, verbose=0)
            y_train = y_train.reshape(
                y_train.shape[0],
            )
            model.fit(X_train, y_train)
            y_preds = model.predict(X_test)
            y_probas = model.predict_proba(X_test)
            metrics = classification_report(y_test, y_preds, output_dict=True)
            results[model.__class__.__name__] = metrics
            wandb.log(metrics)
            wandb.sklearn.plot_classifier(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                y_preds,
                y_probas,
                range(min(y_probas.shape)),
                model_name=name,
                feature_names=None,
            )
            if metrics[self.metric] > best_model[0]:
                best_model[0] = metrics[self.metric]
                best_model[1] = metrics
            wandb.finish()
            dirs = director_exist(os.path.join(os.getenv("MODEL_PATH"), run))
            dump(model, dirs + f"/{name}.joblib")
        return results, best_model[-1]

    def xgb(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        results: Dict = {},
        dimred_technique: str = None,
    ) -> Tuple[Dict[str, Union[str, int]], Dict[str, int]]:
        model = xgb.XGBClassifier(**self.xgb_config)
        name = dimred_technique + model.__class__.__name__
        wandb.init(
            project=PROJECT_NAME,
            name=name,
            config={
                "config": self.xgb_config,
                "model": name,
                "results": results,
                "modelLibrary": "XGB",
            },
        )
        y_train, y_test = label_encoding(y_train, y_test)
        model.fit(
            cp.asarray(X_train),
            cp.asarray(y_train),
            eval_set=[(cp.asarray(X_test), cp.asarray(y_test))],
            callbacks=[WandbCallback(log_model=True)],
        )
        y_preds = model.predict(X_test)
        metrics = classification_report(y_test, y_preds, output_dict=True)
        results[name] = metrics
        wandb.log(metrics)
        wandb.finish()
        dirs = director_exist(os.path.join(os.getenv("MODEL_PATH"), run))
        model.save_model(f"{dirs}/{name}.json")
        return results, metrics

    def lgb(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        results: Dict = {},
        dimred_technique: str = None,
    ) -> Tuple[Dict[str, Union[str, int]], Dict[str, int]]:
        name = dimred_technique + "LGBClf"
        wandb.init(
            project=PROJECT_NAME,
            name=name,
            config={
                "config": self.lgb_config,
                "results": results,
                "modelLibrary": "LGB",
            },
        )
        y_train, y_test = label_encoding(y_train, y_test)
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        model = lgb.train(
            self.lgb_config,
            train_data,
            valid_sets=[test_data],
            callbacks=[wandb_callback()],
        )
        y_preds = model.predict(X_test)
        metrics = classification_report(
            y_test, np.argmax(y_preds, axis=1), output_dict=True
        )
        results[name] = metrics
        log_summary(model, save_model_checkpoint=True)
        wandb.log(metrics)
        wandb.finish()
        dirs = director_exist(os.path.join(os.getenv("MODEL_PATH"), run))
        model.save_model(f"{dirs}/{name}.txt")
        # model.save_model(f'{dirs}/{name}-json.json', format='json')
        return results, metrics

    def evaluate(self) -> Dict[str, Dict[str, Dict[str, Union[str, int]]]]:
        all_pipeline_performance = {}
        outer_iterator = tqdm(self.all_variations)
        best_performances = {
            self.metric: [],
            "pipeline_variation": [],
            # "pipeline_performance": [],
            "pipeline_name": [],
        }
        for pipeline_variation_name in outer_iterator:
            best_performing_pipeline = [0, None, pipeline_variation_name]
            specific_pipeline_variations = self.all_variations[pipeline_variation_name]
            inner_iterator = tqdm(specific_pipeline_variations, leave=False)
            for pipeline_variation in inner_iterator:
                name_of_pipeline = pipeline_variation.__class__.__name__
                pipeline_performance = {}
                X_train = pipeline_variation.fit_transform(self._data["X_train"])
                X_test = pipeline_variation.transform(self._data["X_test"])
                inner_iterator.set_description("Sklearn Model...")
                pipeline_performance, sklearn_metrics = self.sklearn(
                    X_train,
                    X_test,
                    self._data["y_train"],
                    self._data["y_test"],
                    inner_iterator,
                    pipeline_performance,
                    name_of_pipeline,
                )
                inner_iterator.set_description("Sklearn Model Done :)")
                inner_iterator.set_description("XGB Model...")
                pipeline_performance, xgb_metrics = self.xgb(
                    X_train,
                    X_test,
                    self._data["y_train"],
                    self._data["y_test"],
                    pipeline_performance,
                    name_of_pipeline,
                )
                inner_iterator.set_description("XGB Model Done :)")
                inner_iterator.set_description("LGB Model...")
                pipeline_performance, lgb_metrics = self.lgb(
                    X_train,
                    X_test,
                    self._data["y_train"],
                    self._data["y_test"],
                    pipeline_performance,
                    name_of_pipeline,
                )
                inner_iterator.set_description("LGB Model Done :)")
                all_pipeline_performance[name_of_pipeline] = pipeline_performance
                avg_var = average_metric(
                    self.metric, [sklearn_metrics, xgb_metrics, lgb_metrics]
                )
                if avg_var > best_performing_pipeline[0]:
                    best_performing_pipeline[0] = avg_var
                    best_performing_pipeline[1] = str(pipeline_variation).strip
                    # best_performing_pipeline[2] = pipeline_performance
                inner_iterator.set_description(f"{name_of_pipeline} Done :)")
            # best_performances[pipeline_variation_name] = best_performing_pipeline
            best_performances = add_to_dictionary(
                best_performances, best_performing_pipeline
            )
        with open(f'{os.getenv("DATA_PATH")}/all_performance_data.json', "w") as f:
            json.dump(all_pipeline_performance, f)
        with open(
            f'{os.getenv("DATA_PATH")}/best_performance_dimred.json', "w"
        ) as json_f:
            json.dump(best_performances, json_f)
        return all_pipeline_performance, best_performances
