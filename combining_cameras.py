from os import path
import pandas as pd
import numpy as np

def combine_cameras(csv_path, combination_type='mean'):
    df = pd.read_csv(csv_path)
    if combination_type =='similarity':
        return get_cameras_similarity_mean(df)
    elif combination_type =='mean':
        return get_cameras_mean(df)
    else:
        print("combination_type argument is not one of: 'similarity', 'mean'.")

def get_cameras_mean(df):
    # calculates mean for head center location.
    df['finalCenterX'] = df[['topcenterx', 'leftcenterx', 'rightcenterx', 'backcenterx']].mean(axis=1)
    df['finalCenterY'] = df[['topcentery', 'leftcentery', 'rightcentery', 'backcentery']].mean(axis=1)
    df['finalCenterZ'] = df[['topcenterz', 'leftcenterz', 'rightcenterz', 'backcenterz']].mean(axis=1)
    # calculates mean for head direction.
    df['finalDirectionX'] = df[['topdirectionx', 'leftdirectionx', 'rightdirectionx', 'backdirectionx']].mean(axis=1)
    df['finalDirectionY'] = df[['topdirectiony', 'leftdirectiony', 'rightdirectiony', 'backdirectiony']].mean(axis=1)
    df['finalDirectionZ'] = df[['topdirectionz', 'leftdirectionz', 'rightdirectionz', 'backdirectionz']].mean(axis=1)
    return df[['finalCenterX', 'finalCenterY', 'finalCenterZ', 'finalDirectionX', 'finalDirectionY' , 'finalDirectionZ']]

def get_cameras_similarity_mean(df):
    # calculates mean for head center location.
    df['meanCenterX'] = df[['topcenterx', 'leftcenterx', 'rightcenterx', 'backcenterx']].mean(axis=1)
    df['meanCenterY'] = df[['topcentery', 'leftcentery', 'rightcentery', 'backcentery']].mean(axis=1)
    df['meanCenterZ'] = df[['topcenterz', 'leftcenterz', 'rightcenterz', 'backcenterz']].mean(axis=1)

    # for each coordinate, calculates the difference betweeen each camera estimation to a pasudo overall-mean one.
    df['topDiffX'] = df['topcenterx'] - df['meanCenterX']
    df['topDiffY'] = df['topcentery'] - df['meanCenterY']
    df['topDiffZ'] = df['topcenterz'] - df['meanCenterZ']

    df['leftDiffX'] = df['leftcenterx'] - df['meanCenterX']
    df['leftDiffY'] = df['leftcentery'] - df['meanCenterY']
    df['leftDiffZ'] = df['leftcenterz'] - df['meanCenterZ']

    df['rightDiffX'] = df['rightcenterx'] - df['meanCenterX']
    df['rightDiffY'] = df['rightcentery'] - df['meanCenterY']
    df['rightDiffZ'] = df['rightcenterz'] - df['meanCenterZ']

    df['backDiffX'] = df['backcenterx'] - df['meanCenterX']
    df['backDiffY'] = df['backcentery'] - df['meanCenterY']
    df['backDiffZ'] = df['backcenterz'] - df['meanCenterZ']

    # calculates the euclidean distance between each camera's estimation to a pasudo overall-mean one.
    df['topNorm'] = np.linalg.norm(df[['topDiffX', 'topDiffY', 'topDiffZ']].values,axis=1)
    df['leftNorm'] = np.linalg.norm(df[['leftDiffX', 'leftDiffY', 'leftDiffZ']].values,axis=1)
    df['rightNorm'] = np.linalg.norm(df[['rightDiffX', 'rightDiffY', 'rightDiffZ']].values,axis=1)
    df['backNorm'] = np.linalg.norm(df[['backDiffX', 'backDiffY', 'backDiffZ']].values,axis=1)
    # determines a threshold of filtering cameras.
    df['meanNorm'] = df[['topNorm', 'leftNorm', 'rightNorm', 'backNorm']].mean(axis=1)
    df['normSTD'] = df[['topNorm', 'leftNorm', 'rightNorm', 'backNorm']].std(axis=1, ddof=0)
    df['upperBound'] = df['meanNorm'] + df['normSTD']

    # for each camera writes the coordinates data only if the camera's norm didn't pass the thrshold on each frame.
    # X center location
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopCenterX'] = df['topcenterx']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftCenterX'] = df['leftcenterx']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightCenterX'] = df['rightcenterx']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackCenterX'] = df['backcenterx']
    # X head direction
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopDirectionX'] = df['topdirectionx']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftDirectionX'] = df['leftdirectionx']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightDirectionX'] = df['rightdirectionx']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackDirectionX'] = df['backdirectionx']
    # Y center location
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopCenterY'] = df['topcentery']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftCenterY'] = df['leftcentery']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightCenterY'] = df['rightcentery']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackCenterY'] = df['backcentery']
    # Y head direction
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopDirectionY'] = df['topdirectiony']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftDirectionY'] = df['leftdirectiony']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightDirectionY'] = df['rightdirectiony']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackDirectionY'] = df['backdirectiony']
    # Z center location
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopCenterZ'] = df['topcenterz']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftCenterZ'] = df['leftcenterz']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightCenterZ'] = df['rightcenterz']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackCenterZ'] = df['backcenterz']
    # Z head direction
    df.loc[df['topNorm'] <= df['upperBound'], 'filteredTopDirectionZ'] = df['topdirectionz']
    df.loc[df['leftNorm'] <= df['upperBound'], 'filteredLeftDirectionZ'] = df['leftdirectionz']
    df.loc[df['rightNorm'] <= df['upperBound'], 'filteredRightDirectionZ'] = df['rightdirectionz']
    df.loc[df['backNorm'] <= df['upperBound'], 'filteredBackDirectionZ'] = df['backdirectionz']

    # calculates final head center location.
    df['finalCenterX'] = df[['filteredTopCenterX', 'filteredLeftCenterX', 'filteredRightCenterX', 'filteredBackCenterX']].mean(axis=1)
    df['finalCenterY'] = df[['filteredTopCenterY', 'filteredLeftCenterY', 'filteredRightCenterY', 'filteredBackCenterY']].mean(axis=1)
    df['finalCenterZ'] = df[['filteredTopCenterZ', 'filteredLeftCenterZ', 'filteredRightCenterZ', 'filteredBackCenterZ']].mean(axis=1)
    # calculates final head direction.
    df['finalDirectionX'] = df[['filteredTopDirectionX', 'filteredLeftDirectionX', 'filteredRightDirectionX', 'filteredBackDirectionX']].mean(axis=1)
    df['finalDirectionY'] = df[['filteredTopDirectionY', 'filteredLeftDirectionY', 'filteredRightDirectionY', 'filteredBackDirectionY']].mean(axis=1)
    df['finalDirectionZ'] = df[['filteredTopDirectionZ', 'filteredLeftDirectionZ', 'filteredRightDirectionZ', 'filteredBackDirectionZ']].mean(axis=1)

    return df[['finalCenterX', 'finalCenterY', 'finalCenterZ', 'finalDirectionX', 'finalDirectionY' , 'finalDirectionZ']]