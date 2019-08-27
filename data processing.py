import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from pdb import set_trace

def preprocess(features):
    new_feas = features.copy()
    # Clean data
    # complete missing age value with median
    new_feas["Age"].fillna(new_feas["Age"].median(),inplace=True)
    # complete embarked with mode
    new_feas["Embarked"].fillna(new_feas["Embarked"].mode()[0],inplace=True)
    # complete missing fare with median
    new_feas["Fare"].fillna(new_feas["Fare"].median(),inplace=True)
    # Drop unused columns
    drop_columns = ["PassengerId","Cabin","Ticket"]
    new_feas.drop(drop_columns,axis=1,inplace=True)

    # Feature engineer
    # Compute the size of family of each individual
    new_feas["FamilySize"] = new_feas["SibSp"]+new_feas["Parch"]+1
    new_feas["IsAlone"] = 1
    new_feas["IsAlone"].loc[new_feas["FamilySize"]>1]=0
    new_feas["Title"] = new_feas["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
    # Using quantiles to cut Fare into equal size bins
    # indicating different level of fares
    new_feas["FareBin"] = pd.qcut(new_feas["Fare"],4)
    # linearly cut age into ranges
    # indicating different age stage
    new_feas["AgeBin"] = pd.cut(new_feas["Age"].astype(int),5)

    # Clean up rare title names
    # Statistically minimum
    stat_min = 10
    # For title names that less than 10 people have it
    # we will deem it as rare and categorize it into Misc titles
    title_names = (new_feas["Title"].value_counts()<stat_min)
    new_feas["Title"]=new_feas["Title"].apply(lambda x:"Misc" if title_names.loc[x]==True else x)

    label = LabelEncoder()
    new_feas["Sex_Code"] = label.fit_transform(new_feas["Sex"])
    new_feas["Embarked_Code"]=label.fit_transform(new_feas["Embarked"])
    new_feas["Title_Code"]=label.fit_transform(new_feas["Title"])
    new_feas["AgeBin_Code"]=label.fit_transform(new_feas["AgeBin"])
    new_feas["FareBin_Code"]=label.fit_transform(new_feas["FareBin"])

    return new_feas

def preprocess2(train_data,test_data):
    age_median = train_data["Age"].median()
    embarked_mode = train_data["Embarked"].mode()[0]
    fare_median = train_data["Fare"].median()

    # Clean data
    # complete missing age value with median
    train_data["Age"].fillna(age_median,inplace=True)
    test_data["Age"].fillna(age_median,inplace=True)
    # complete embarked with mode
    train_data["Embarked"].fillna(embarked_mode,inplace=True)
    test_data["Age"].fillna(embarked_mode,inplace=True)
    # complete missing fare with median
    train_data["Fare"].fillna(fare_median,inplace=True)
    test_data["Fare"].fillna(fare_median,inplace=True)

    # # Drop unused columns
    # drop_columns = ["PassengerId","Cabin","Ticket"]
    # train_data.drop(drop_columns,axis=1,inplace=True)
    # test_data.drop(drop_columns,axis=1,inplace=True)

    # Feature engineer
    # Compute the size of family of each individual
    # For train data
    train_data["FamilySize"] = train_data["SibSp"]+train_data["Parch"]+1
    train_data["IsAlone"] = 1
    train_data["IsAlone"].loc[train_data["FamilySize"]>1]=0
    train_data["Title"] = train_data["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]
    # For test data
    test_data["FamilySize"] = test_data["SibSp"]+test_data["Parch"]+1
    test_data["IsAlone"] = 1
    test_data["IsAlone"].loc[test_data["FamilySize"]>1]=0
    test_data["Title"] = test_data["Name"].str.split(", ",expand=True)[1].str.split(".",expand=True)[0]

    # Using quantiles to cut Fare into equal size bins
    # indicating different level of fares
    train_data["FareBin"] = pd.qcut(train_data["Fare"],4)
    farebin_categories = train_data["FareBin"].cat.categories
    test_data["FareBin"] = pd.cut(test_data["Fare"],farebin_categories)
    # In case there are people in test set spent more fare than those in train set
    biggest_farebin = farebin_categories[3]
    test_data["FareBin"].fillna(biggest_farebin,inplace=True)

    # linearly cut age into ranges
    # indicating different age stage
    train_data["AgeBin"] = pd.cut(train_data["Age"].astype(int),5)
    agebin_categories = train_data["AgeBin"].cat.categories
    test_data["AgeBin"] = pd.cut(test_data["Age"],agebin_categories)

    # Clean up rare title names
    # Statistically minimum
    stat_min = 10
    # For title names that less than 10 people have it
    # we will deem it as rare and categorize it into Misc titles
    title_names = (train_data["Title"].value_counts()<stat_min)
    train_data["Title"]=train_data["Title"].apply(lambda x:"Misc" if title_names.loc[x]==True else x)
    test_data["Title"]=test_data["Title"].apply(lambda x:"Misc" if x not in title_names or title_names.loc[x]==True else x)

    # Encode categorical data into integer so that we can perform math operations
    # For train data
    sex_encoder = LabelEncoder()
    embarked_encoder = LabelEncoder()
    title_encoder =LabelEncoder()
    agebin_encoder = LabelEncoder()
    farebin_encoder = LabelEncoder()
    train_data["Sex_Code"] = sex_encoder.fit_transform(train_data["Sex"])
    train_data["Embarked_Code"]=embarked_encoder.fit_transform(train_data["Embarked"])
    train_data["Title_Code"]=title_encoder.fit_transform(train_data["Title"])
    train_data["AgeBin_Code"]=agebin_encoder.fit_transform(train_data["AgeBin"])
    train_data["FareBin_Code"]=farebin_encoder.fit_transform(train_data["FareBin"])
    # For test data
    test_data["Sex_Code"] = sex_encoder.transform(test_data["Sex"])
    test_data["Embarked_Code"]=embarked_encoder.transform(test_data["Embarked"])
    test_data["Title_Code"]=title_encoder.transform(test_data["Title"])
    test_data["AgeBin_Code"]=agebin_encoder.transform(test_data["AgeBin"])
    # set_trace()
    test_data["FareBin_Code"]=farebin_encoder.transform(test_data["FareBin"])

if __name__=="__main__":
    f = open("../Data/train.csv")
    train_df = pd.read_csv(f)
    f.close()
    f = open("../Data/test.csv")
    test_df = pd.read_csv(f)
    f.close()
    # Preprocess
    # new_train_df = preprocess(train_df)
    # new_train_df.to_csv("../Data/preprocessed_train.csv")
    preprocess2(train_df,test_df)
    train_df.to_csv("../Data/preprocessed_train.csv",index=False)
    test_df.to_csv("../Data/preprocessed_test.csv",index=False)