def test_import():
    from carbondriver import GDEOptimizer

    gde = GDEOptimizer(output_dir="./tmp_test_out")

def test_gde_optimizer_within():
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_data

    import pandas as pd
    
    gde = GDEOptimizer("MLP", output_dir="./tmp_test_out")

    _, _, _, _, df = load_data("paper/Characterization_data.xlsx")

    df_train = df.loc[:30]
    df_explore = df.loc[31:]

    ei, next_pick = gde.step_within_data(df_train, df_explore)

    print("First pick:", ei, int(next_pick))
    
    df_new =  df_explore.iloc[int(next_pick)]
    df_explore =  df_explore.drop(index=df_new.name)

    ei, next_pick = gde.step_within_data(df_new, df_explore)

    print("Second pick", ei, int(next_pick))

def test_gde_optimizer_free():
    from carbondriver import GDEOptimizer
    from carbondriver.loaders import load_data
    
    gde = GDEOptimizer("MLP", output_dir="./tmp_test_out")

    _, _, _, _, df = load_data("paper/Characterization_data.xlsx")

    ei, next_pick = gde.step(df)

    print(ei, next_pick)

    
