import pandas as pd
import numpy as np

def clean_users(users_df):
    """
    Takes the user dataframe and cleans it according to the steps determined during exploration.
    1. Income fillna(0) to allow for a cut of "unknown" for 0-30k income
    2. Cut income into segments (0-30k, 30-50k, 50-80k, over 80k)
    3. Replace NaN codes in remainding variables with "unknown" to give it a categorical level in those variables
    4. Cut age into segments (under 40, 40-60, 60-80, over 80)
    5. Remove became_member_on.
    6. Remove income (continuous version).
    """
    # Fill income NaN with 0
    users_df.income.fillna(0, inplace = True)

    # Cut income into segments
    users_df['income_segment'] = pd.cut(users_df.income,
                                        bins = [0,30000,50000,80000,200000],
                                        labels = ['Unknown','Under 50k','50k-80k','Over 80k'],
                                        include_lowest = True)

    # Cut age into segments
    users_df['age_segment'] = pd.cut(users_df.age,
                                    bins = [0,24,40,60,80,117,200],
                                    labels = ['24 and Under','25-40','41-60','61-80','81 and Over','Unknown'],
                                    include_lowest = False)

    # Replace NaN codes in gender with "unknown" and drop gender "O"
    users_df.loc[users_df['gender'].isnull(), 'gender'] = 'Unknown'
    users_df = users_df.loc[users_df['gender'] != 'O']

    # Drop income, became_member_on and age columns
    clean_users = users_df.drop(columns = ['income','age','became_member_on'])

    return clean_users

def clean_transactions(trans_df, offer_df, users_df):
    """
    Takes the transactions dataframe and cleans it. Turns it into a multi-outcome array that plays
    nicely with the results of a cluster analysis performed on the users dataframe.

    Ultimate goal is seeing if cluster representation changes for each offer ID.

    Final frame represents initial response rates to each offer (first receipt). Once we cluster the users dataframe,
    we can merge it with this frame and then show response rates by cluster for each offer. 1 means the user completed the
    offer within the offer's duration (from the time of receipt).

    One step further, we can also merge offers to get offer type.
    Then break down response rates by cluster for each offer type.
    """
    # Filter df for important events, only (events with event ID attached, bc all we care about is response or non-response)
    sig_events = ['offer received', 'offer viewed', 'offer completed']
    filtered_df = trans_df.loc[trans_df['event'].isin(sig_events)].reset_index(drop = True).copy()

    # Create offer_id variable from value
    # NOTE: value is a series of dict. Offer ID key when event is viewed is "offer id." It's "offer_id" when event is completed.
    filtered_df['offer_id'] = filtered_df['value'].apply(
        lambda x: x['offer_id'] if 'offer_id' in x.keys() else x['offer id']
    )
    filtered_df.drop(columns = ['value'],
                     inplace = True)

    # Deduplicate the dataframe for person, event and offer_id. Take min(time) for each.
    deduped_df = filtered_df.groupby(['person','event','offer_id'],
                                     as_index = False)['time'].agg(np.min)

    # Pivot out event with time as the values to see when each person received and completed each offer
    # Resetting index leaves person and offer_id as columns rather than creating a mult-index from them
    # Will make merging easier, later
    pivoted_df = deduped_df.pivot(index = ['person','offer_id'],
                                  columns = ['event'],
                                  values = 'time').reset_index(drop = False)

    # Filter pivoted_df so that we're only working with valid test cases for response rates
    # Must be a viewed time.
    # Completed time can't be less than viewed time (greater than and NaN both okay)
    # Days between view and receipt must <= the offer term in days
    filtered_pivot = pivoted_df.loc[
        (~pivoted_df['offer viewed'].isnull()) &
        (~(pivoted_df['offer completed'] < pivoted_df['offer viewed']))
    ]
    filtered_pivot = filtered_pivot.merge(
        offer_df[['id','duration','offer_type']],
        how = 'left',
        left_on = 'offer_id',
        right_on = 'id'
    ).drop(columns = 'id')
    # Convert offer viewed and offer completed (hours) to days
    filtered_pivot[['offer received','offer viewed','offer completed']] = filtered_pivot[['offer received','offer viewed', 'offer completed']] / 24

    # Add a column for response, 1 if completion happened in offer window. Else 0.
    filtered_pivot['offer_response'] = filtered_pivot.apply(
        lambda x: 1 if x['offer completed'] - x['offer received'] <= x['duration'] else 0,
        axis = 1
    )

    # filter the final product for only person that appears in users_clean
    filtered_pivot = filtered_pivot.loc[filtered_pivot['person'].isin(users_df['id'].unique())].reset_index(drop = True).copy()

    return filtered_pivot[['person','offer_id','offer_type','offer_response']]
