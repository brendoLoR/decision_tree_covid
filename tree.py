# Load TF-DF
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

task = tfdf.keras.Task.CLASSIFICATION
model = tfdf.keras.RandomForestModel(task=task)
target = "OBITO"
porcentagem_obito = 0


def train():
    # Load a dataset in a Pandas dataframe.
    df = pd.read_csv('datasets/COVID.CSV', sep=';')
    # cols_to_drop = ["DATE_DIED"]
    cols_to_drop = ["SEX",  "RENAL_CHRONIC", "CARDIOVASCULAR", "ASTHMA",
                    "OTHER_DISEASE", "COPD", "INMSUPR", "DATE_DIED", "TOBACCO",
                    "PREGNANT", "USMER", "DIABETES", "HIPERTENSION"]

    df = df.drop(columns=cols_to_drop)

    # drop all dolumns with 97 and 99
    df['CLASIFFICATION_FINAL'] = np.where(df['CLASIFFICATION_FINAL'] > 3, 2, 1)
    for label in df.head():
        df = df[df[label] != 97]
        df = df[df[label] != 99]
        if (label != "AGE"):
            df[label] = np.where(df[label] == 2, 0, 1)

    df_died = df[(df.OBITO == 1)]

    porcentagem_obito = (len(df_died)/len(df)) * 100

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(df.AGE, 15, rwidth=0.9)
    axs[0, 0].hist(df_died.AGE, 15, rwidth=0.9)
    axs[0, 0].grid(axis='y', alpha=0.75)
    axs[0, 0].set_title("Idade")

    axs[1, 0].hist(df.CLASIFFICATION_FINAL, 2,
                   rwidth=0.9, orientation="horizontal")
    axs[1, 0].hist(df_died.CLASIFFICATION_FINAL, 2,
                   rwidth=0.9, orientation="horizontal")
    axs[1, 0].grid(axis='x', alpha=0.75)
    axs[1, 0].set_title("Resultado de teste para COVID")

    axs[0, 1].hist(df.PNEUMONIA, 2, rwidth=0.9, orientation="horizontal")
    axs[0, 1].hist(df_died.PNEUMONIA, 2, rwidth=0.9, orientation="horizontal")
    axs[0, 1].grid(axis='x', alpha=0.75)
    axs[0, 1].set_title("Pacientes com pneumonia")

    axs[1, 1].hist(df.INTUBED, 2, rwidth=0.9, orientation="horizontal")
    axs[1, 1].hist(df_died.INTUBED, 2, rwidth=0.9, orientation="horizontal")
    axs[1, 1].grid(axis='x', alpha=0.75)
    axs[1, 1].set_title("Pacientes que foram entubados")
    fig.show()

    train_df, test_df = train_test_split(df, test_size=0.2,
                                         random_state=42,
                                         shuffle=True)

    # Convert the dataset into a TensorFlow dataset.

    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        train_df, label=target, task=task)
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_df, label=target, task=task)

    # Train a Random Forest model.

    # optional step - add evaluation metrics
    model.compile(metrics=["mse", "mape", "accuracy"])

    model.fit(train_ds)

    # Evaluate the model.
    evaluation = model.evaluate(test_ds, return_dict=True)

    print(evaluation)
    print(f"MSE: {evaluation['mse']:.2f}")
    print(f"RMSE: {math.sqrt(evaluation['mse']):.2f}")
    print(f"MAPE: {evaluation['mape']:.2f}")
    print(f"accuracy: {evaluation['accuracy']:.2f}")

    # Export the model to a SavedModel.


def model_info():
    # Summary of the model structure.
    print("plotting summary")
    model.summary()

    # plot the first tree, restricted to depth of 3
    print("plotting tree")
    html = tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=15)

    output_file = open('index.html', 'w')
    output_file.write(html)
    output_file.close()

    # print("plotting graphic")
    # model.make_inspector().features()
    # model.make_inspector().evaluation()

    inspector = model.make_inspector()
    print("Model type:", inspector.model_type())

    print("Number of trees:", inspector.num_trees())
    print("Objective:", inspector.objective())
    print("Input features:", inspector.features())

    inspector.evaluation()

    print(f"Available variable importances:")
    for importance in inspector.variable_importances().keys():
        print("\t", importance)
    # inspector.variable_importances()["SUM_SCORE"]

    logs = inspector.training_logs()

    plt.figure(figsize=(12, 4))

    # Mean decrease in AUC of the class 1 vs the others.
    variable_importances = inspector.variable_importances()["SUM_SCORE"]

    # Extract the feature name and importance values.
    #
    # `variable_importances` is a list of <feature, importance> tuples.
    feature_names = [vi[0].name for vi in variable_importances]
    feature_importances = [vi[1] for vi in variable_importances]
    # The feature are ordered in decreasing importance value.
    feature_ranks = range(len(feature_names))

    bar = plt.barh(feature_ranks, feature_importances,
                   label=[str(x) for x in feature_ranks])
    plt.yticks(feature_ranks, feature_names)
    plt.gca().invert_yaxis()

    # TODO: Replace with "plt.bar_label()" when available.
    # Label each bar with values
    for importance, patch in zip(feature_importances, bar.patches):
        plt.text(patch.get_x() + patch.get_width(),
                 patch.get_y(), f"{importance:.4f}", va="top")

    plt.xlabel("SUM_SCORE")
    plt.title("Soma do score de cada vari??vel")
    plt.tight_layout()
    plt.show()

    # explain the model's predictions using SHAP

    plt.figure(figsize=(12, 4))

    plt.plot([log.num_trees for log in logs], [
        log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("RMSE (out-of-bag)")
    plt.title("RMSE vs number of trees")

    plt.show()


train()
model_info()
