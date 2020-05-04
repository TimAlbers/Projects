
def describe_nan(df):
    """
    Return transposed pandas describe with share of nan-values

    Parameters:
    df = pandas.DataFrame
    """
    df_des = df.describe().transpose()
    df_des["share_nan"] = 1-(df_des["count"]/df.shape[0])
    df_des_col_order = ["share_nan", "count", "mean",
                        "std", "min", "25%", "50%", "75%", "max"]
    df_des = df_des[df_des_col_order]
    return df_des


def describe_frequent(df, n=5):
    """
    Returns dataframe with most common values

    Parameters:
    df = pandas.DataFrame
    n = Optional number of most common values

    """
    topn_df = pd.DataFrame()
    col_names = df.columns

    for col in col_names:
        col_name = df[col].name
        top_n = df[col].value_counts(dropna=False)[
            :n].reset_index().reset_index()
        top_n_norm = df[col].value_counts(dropna=False, normalize=True)[
            :n].reset_index().reset_index()
        top_n["norm"] = (top_n_norm[col_name]*100).astype(int)
        top_n["level_0"] = top_n["level_0"]+1
        top_n["string"] = "value: "+top_n["index"].astype(
            str)+" | count: "+top_n[col_name].astype(str)+" | norm %: "+top_n["norm"].astype(str)
        result = top_n["string"]
        result.rename(col_name, inplace=True)
        topn_df = pd.concat([topn_df, result], axis=1)
    topn_df.index = "Top"+(topn_df.index + 1).astype(str)
    return topn_df

# main function


def describe_plus(df, subset=None, output=None,
                  figsize=(15, 20), cmap="RdBu"):
    '''
    Returns a dataframe of descripitive strings.
    If only a dataframe is passed, pandas 'describe'-method includes missing value count in %.
    If also one subset is passed, the resulting subset is compared to the entire dataset.

    Parameters:
    df: Required pandas.DataFrame
    subset: Optional pandas.DataFrame with identical columns to compare to df
    output: Optional, if only one df is passed, "non_num" returns df with all
            non numeric columns and adds descriptive stats as first rows.
            If subset is passed, "clean" returns a DataFrame without styling.
            If subset is passed, "sorted" returns a styled DataFrame,
            sorted by z-score. 
    cmap: Optional parameter, default colormap "RdBu"

    Returns:
    (styled) pandas.Dataframe
    '''

    if subset is None:
        if output == "non_num":
            # get only the numeric columns from the dataframe
            all_cols = df.columns
            # describe only works on numerical columns
            num_cols = describe_nan(df).index
            non_num_cols = [col for col in all_cols if col not in num_cols]
            df_non_num = df[non_num_cols]

            # calc some stats on non-numeric columns
            df_stats = pd.DataFrame(columns=non_num_cols)
            df_stats.loc["Record count"] = df_non_num.shape[0]
            df_stats.loc["Count Nan"] = df_non_num.isna().sum()
            df_stats.loc["Share Nan"] = df_non_num.isna().sum() / \
                df_non_num.shape[0]
            df_stats.loc["Count unique"] = df_non_num.nunique()

            # add most common values
            df_stats = pd.concat([df_stats, describe_frequent(df_non_num)])

            # prepend stats to df
            df_non_num_up = pd.concat([df_stats, df_non_num])

            return df_non_num_up

        else:
            return describe_nan(df).style.background_gradient(subset="share_nan", low=0, high=1, cmap="Blues")

    else:
        # calculate descriptive stats
        subset_desc = describe_nan(subset)
        df_desc = describe_nan(df)

        # calculate percent change (used to be for sns.heatmap)
        hm_data = (subset_desc/df_desc)
        hm_data["z-score, subset mean"] = (subset_desc["mean"] -
                                           df_desc["mean"])/(df_desc["std"])
        hm_data["count"] = 0  # no color for count
        hm_data = hm_data[['count', "share_nan", 'z-score, subset mean',
                           'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        # format main annotation values
        main_values = pd.DataFrame()
        main_values = pd.concat(
            [main_values, subset_desc.loc[:, "mean":"max"]], axis=1)
        main_values = main_values.applymap('{:,.2f}'.format)
        main_values["z-score, subset mean"] = (
            subset_desc["mean"]-df_desc["mean"])/(df_desc["std"])
        main_values["count"] = subset_desc['count'].apply('{:,.0f}'.format)
        main_values["share_nan"] = (
            subset_desc['share_nan']*100).apply('{:,.0f}%'.format)
        main_values = main_values[[
            'count', "share_nan", 'z-score, subset mean', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

        # values of subset are compared to main dataset in annotation as follows:
        # count (share of total records)
        # share_Nan (% compared to entire dataset)
        # mean (% compared to entire dataset)
        # z-score of subset-mean (no comparison)
        # std (% compared to entire dataset)
        # min,25%,50%,75%,max (percentile rank of entire dataset)
        # dataframe has same columns, but z-score is also seperate and shaded
        # percentages are formatted as integer with sign

        # pct rank: copy output from description method to original dataframe
        subset_desc_pct_rank = pd.concat(
            [df, subset_desc.loc[:, "min":"max"].transpose()]).dropna(axis=1, subset=["min"])

        # pct rank: make a column with percentile rank for every numeric column
        list_col_pct = []
        for col in subset_desc_pct_rank:
            col_pct = col+"_pct_rank"
            list_col_pct.append(col_pct)
            subset_desc_pct_rank[col_pct] = subset_desc_pct_rank[col].rank(
                pct=True)

        # pct rank: discard main dataset, apply proper names and shape
        subset_desc_pct_rank = (
            subset_desc_pct_rank.loc["min":"max", list_col_pct].transpose()*100).astype(int)
        subset_desc_pct_rank.index = subset_desc_pct_rank.index.str.replace(
            "_pct_rank", "")
        subset_desc_pct_rank.columns = subset_desc_pct_rank.columns+"_pct_rank"

        # format all values going into parenthesis
        paran = pd.DataFrame()
        paran["count"] = ((subset_desc/df_desc)["count"]
                          * 100).apply('{:,.0f}%'.format)
        paran["share_nan"] = hm_data["share_nan"].apply('{:,.0f}%'.format)
        paran["z-score, subset mean"] = "-"
        paran["mean"] = ((((subset_desc/df_desc)["mean"])-1)
                         * 100).apply('{:+,.0f}%'.format)
        paran["std"] = ((((subset_desc/df_desc)["std"])-1)
                        * 100).apply('{:+,.0f}%'.format)
        paran = pd.concat([paran, subset_desc_pct_rank], axis=1)
        paran.columns = paran.columns.str.replace("_pct_rank", "")

        # build the annotation
        annot = main_values.astype(str)+" ("+paran.astype(str)+")"
        annot["z-score, subset mean"] = main_values["z-score, subset mean"]

        # decided to redo column headers
        new_col_headers = {'count': "count (share of df)",
                           "share_nan": "share_nan (change to df)",
                           'z-score, subset mean': "z-score, subset mean",
                           'mean': " mean (change to df)",
                           'std': "std (change to df)",
                           'min': "min (pct rank in df)",
                           '25%': "25% (pct rank in df)",
                           '50%': "50% (pct rank in df)",
                           '75%': "75% (pct rank in df)",
                           'max': "max (pct rank in df)"
                           }
        annot.rename(inplace=True, columns=new_col_headers)

        if output == "clean":
            return annot

        elif output == "sorted":
            return annot.sort_values(by="z-score, subset mean").style.background_gradient(
                subset="z-score, subset mean", cmap="RdBu")

        else:
            return annot.style.background_gradient(
                subset="z-score, subset mean", cmap="RdBu")