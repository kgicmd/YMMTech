import os

import pandas as pd


def read_csv(path, strip=False):
    res = pd.read_csv(path, skipinitialspace=strip)
    return res


def flat_map(df, key_col, value_col):
    df = df.set_index([key_col])
    new_df = df[value_col].apply(pd.Series).unstack().reset_index().dropna()
    new_df = new_df.iloc[:, 1:]
    new_df = new_df.rename(columns={new_df.columns[0]: 'user_id', new_df.columns[1]: value_col})
    return new_df


def make_original_df(
        driver_action: pd.DataFrame,
        driver_portrait: pd.DataFrame,
        cargo_record: pd.DataFrame,
        city_distance: pd.DataFrame,
        is_train=True):
    """
    合并相关的表（司机画像，司机行为，发货记录）。
    :param city_distance: 城市间距离
    :param driver_action: 司机行为日志
    :param driver_portrait: 司机画像
    :param cargo_record: 发货记录
    :param is_train: 是否训练集
    :return:
    """
    print('Making original DataFrame...')
    res_df = driver_action
    # Join driver portrait.
    res_df = res_df.merge(driver_portrait, on='user_id', how='left')
    res_df = res_df.fillna(method='ffill')

    # Join city_distance.
    city_distance_exc = city_distance.rename(
        columns={'start_city_id': 'end_city_id', 'end_city_id': 'start_city_id'})
    city_distance = pd.concat([city_distance, city_distance_exc], sort=False)
    city_distance = city_distance.drop_duplicates(subset=['start_city_id', 'end_city_id'])

    city_distance['start_province'] = city_distance.start_city_id // 10000
    city_distance['end_province'] = city_distance.end_city_id // 10000

    dist_mean = city_distance.groupby(['start_province', 'end_province']).distance.mean().reset_index()
    dist_mean = dist_mean.rename(columns={'distance': 'mean_distance'})

    cargo_record = cargo_record.drop_duplicates(subset=['id'])
    cargo_record = cargo_record.merge(city_distance, how='left', on=['start_city_id', 'end_city_id'])
    cargo_record['start_province'] = cargo_record.start_city_id // 10000
    cargo_record['end_province'] = cargo_record.end_city_id // 10000
    cargo_record = cargo_record.merge(dist_mean, on=['start_province', 'end_province'], how='left')
    cargo_record.distance.fillna(cargo_record.mean_distance, inplace=True)
    # Join cargo record.
    res_df = res_df.merge(cargo_record, on='id', how='left', suffixes=['_driver', '_cargo'])

    if is_train:
        # 丢弃没有货物信息的样本
        res_df = res_df.dropna(subset=['shipper_id'])

    select_cols = ['user_id', 'gender', 'id', 'install_city_short', 'install_prov_short', 'truck_type_driver',
                   'truck_len', 'regular_subscribe_line', 'line', 'line_30days', 'line_60days',
                   'regular_search_line_3', 'regular_search_line_7', 'regular_search_line_14',
                   'regular_search_line_30',
                   'scan_cargo_cnt_3', 'scan_cargo_cnt_7', 'scan_cargo_cnt_14', 'scan_cargo_cnt_30',
                   'click_cargo_cnt_3', 'click_cargo_cnt_7', 'click_cargo_cnt_14', 'click_cargo_cnt_30',
                   'call_cnt_3', 'call_cnt_7', 'call_cnt_14', 'call_cnt_30', 'call_cnt_45',
                   'call_cargo_cnt_3', 'call_cargo_cnt_7', 'call_cargo_cnt_14', 'call_cargo_cnt_30',
                   'call_cargo_cnt_45', 'open_days_3', 'open_days_7', 'open_days_14', 'open_cnt_3',
                   'open_cnt_7', 'open_cnt_14', 'call_cnt_3', 'call_cnt_7', 'call_cnt_14', 'call_days_3',
                   'call_days_7', 'call_days_14', 'order_days_3', 'order_days_7', 'order_days_14',
                   'order_cnt_3', 'order_cnt_7', 'order_cnt_14', 'distance',
                   'shipper_id', 'start_city_id', 'end_city_id', 'truck_length', 'cargo_capacipy', 'cargo_type',
                   'handling_type', 'highway', 'expect_freight', 'truck_type_list', 'truck_type_cargo',
                   'truck_count', 'mileage', 'lcl_cargo']
    if is_train:
        select_cols.append('label')

    res_df = res_df[select_cols]
    res_df = res_df.rename(columns={'truck_len': 'truck_len_driver', 'truck_length': 'truck_len_cargo',
                                    'cargo_capacipy': 'cargo_capacity'})
    return res_df


def _is_length_match(truck_len: pd.Series,
                     match_target: pd.Series):
    """
    司机画像中的车长是否符合要求。
    :param truck_len: 司机画像中的车长信息
    :param match_target: 发货记录中的车长需求
    :return: pd.Series
    车长是否符合要求
    """
    match_target = pd.Series(match_target.astype(str))
    match_target = match_target.fillna('')
    match_target = match_target.str.split(',')
    match_target = match_target.apply(lambda x: [float(e) for e in x])
    tmp = pd.DataFrame({'truck_len': truck_len, 'target': match_target})
    return tmp.apply(lambda x: x['truck_len'] >= min(x['target']), axis=1)


def _is_line_match(start_end_province: pd.Series,
                   match_target: pd.Series):
    """
    司机画像中的常跑线路和订阅线路是否包含目标起止省份。
    :param start_end_province: 目标省份
    :param match_target: 司机画像中的线路信息
    :return: pd.Series
    是否司机感兴趣的线路
    """
    print('Adding `%s`...' % match_target.name)
    match_target = match_target.fillna('').str.replace('0', '')
    match_target = match_target.str.split(',')
    tmp = pd.DataFrame({'start_end': start_end_province, 'target': match_target})
    return tmp.apply(lambda x: x['start_end'] in x['target'], axis=1)


def make_features_df(ori: pd.DataFrame, is_train=True, bins_path=''):
    """
    构造特征
    :param is_train: 是否是训练集
    :param ori: 连接好的包含用户画像和货物信息的表
    :param bins_path: 训练集持久化保存的且分点路径
    :return: pd.DataFrame
    构造好的特征
    """
    print('Making features DataFrame...')

    res_df = ori.loc[:, ['user_id', 'id', 'label']]

    # gender
    print('Adding `gender`...')
    res_df['gender'] = ori.gender.map({'女': 0, '男': 1, '未知': -1})

    # install_city_short match
    print('Adding `install_city_match`...')
    res_df['install_city_match'] = ori.apply(
        lambda row:
        row['install_city_short'] == row['start_city_id'] or
        row['install_city_short'] == row['end_city_id'], axis=1)

    # cargo_type, handling_type, truck_count, highway, lcl_cargo
    res_df['cargo_type'] = ori.cargo_type.iloc[:].fillna(-1)
    res_df['handling_type'] = ori.handling_type.iloc[:].fillna(0)

    need_count = ori.truck_len_cargo.astype(str).fillna('').str.split(',').apply(len)
    ori.loc[ori.truck_count <= 0, 'truck_count'] = need_count.loc[ori.truck_count <= 0]
    ori.loc[ori.truck_count <= 0, 'truck_count'] = 1
    res_df['truck_count'] = ori.truck_count.fillna(1)

    res_df['highway'] = ori.highway.loc[:].fillna(0)
    res_df['lcl_cargo'] = ori.lcl_cargo.loc[:].fillna(0)

    # Add truck_type related features.
    print('Adding `truck_type_cargo_match*` features...')
    truck_type_driver_order = {}  # key: truck_type_driver, value: truck_type_cargo
    for truck_type in set(ori.truck_type_driver):
        truck_type_driver_order[truck_type] = \
            ori.loc[ori.truck_type_driver == truck_type, ['truck_type_driver', 'truck_type_cargo']][
                'truck_type_cargo'].value_counts().index[:4]

    for i, order in enumerate(['first', 'second', 'third', 'fourth']):
        target_field = 'truck_type_cargo_match_%s' % order
        res_df[target_field] = ori.apply(
            lambda row:
            row['truck_type_driver'] in truck_type_driver_order and
            i < len(truck_type_driver_order[row['truck_type_driver']]) and
            row['truck_type_cargo'] == truck_type_driver_order[row['truck_type_driver']][i], axis=1)

    print('Adding `truck_type_driver_match*` features...')
    truck_type_cargo_order = {}  # key: truck_type_cargo, value: truck_type_driver
    for truck_type in set(ori.truck_type_cargo):
        truck_type_cargo_order[truck_type] = \
            ori.loc[ori.truck_type_cargo == truck_type, ['truck_type_driver', 'truck_type_cargo']][
                'truck_type_driver'].value_counts().index[:4]

    for i, order in enumerate(['first', 'second', 'third', 'fourth']):
        target_field = 'truck_type_driver_match_%s' % order
        res_df[target_field] = ori.apply(
            lambda row:
            row['truck_type_cargo'] in truck_type_cargo_order and
            i < len(truck_type_cargo_order[row['truck_type_cargo']]) and
            row['truck_type_driver'] == truck_type_cargo_order[row['truck_type_cargo']][i], axis=1)

        # Add truck_len_match feature
        print('Adding `truck_len_match`...')
        res_df['truck_len_match'] = _is_length_match(ori.truck_len_driver, ori.truck_len_cargo)

        # Add line match related features.
        ori['start_city_id'] = ori.start_city_id.astype(str)
        ori['end_city_id'] = ori.end_city_id.astype(str)
        start_end = ori.start_city_id.str[:2] + '-' + ori.end_city_id.str[:2]

        # Add regular_subscribe_match feature.
        res_df['regular_subscribe_match'] = _is_line_match(start_end, ori.regular_subscribe_line)
        res_df['regular_subscribe_match'].fillna(False)

        # Add regular_search related features.
        for days in [3, 7, 14, 30]:
            target_field = 'regular_search_%d_match' % days
            ori_field = 'regular_search_line_%d' % days
            res_df[target_field] = _is_line_match(start_end, ori[ori_field])

        # line_match features.
        for sufix in ['', '_30days', '_60days']:
            target_field = 'line%s_match' % sufix
            ori_field = 'line%s' % sufix
            res_df[target_field] = _is_line_match(start_end, ori[ori_field])
            res_df[target_field].fillna(False)

        # Cargo related counts.
        bins = 10
        for action in ['scan', 'click', 'call']:
            avgs = {}
            for days in [3, 7, 14, 30]:
                target_field = '%s_cargo_level_%d' % (action, days)
                ori_field = '%s_cargo_cnt_%d' % (action, days)
                print('Adding `%s`...' % target_field)
                data: pd.Series = ori[ori_field].fillna(0)
                avgs[days] = data.values // days
                res_df[target_field] = pd.cut(data, bins=bins, labels=range(bins))
            avg_filed = '%s_cargo_level_avg' % action
            print('Adding `%s`...' % avg_filed)
            res_df[avg_filed] = avgs[7]

        # # other counts.
        # prefixes = ['open_days', 'open_cnt', 'call_days', 'call_cnt',
        #             'order_days', 'order_cnt']
        #
        # if is_train:
        #     cut_bins = {}
        # else:
        #     # Load persisted `cut_bins` obj.
        #     if os.path.exists(bins_path):
        #         cut_bins = pickle.load(open(bins_path, mode='rb'))
        #     else:
        #         raise FileNotFoundError('File `%s` not found.' % bins_path)
        #
        # for i, prefix in enumerate(prefixes):
        #     for days in [3, 7, 14]:
        #         ori_field = '%s_%d' % (prefix, days)
        #         target_field = '%s_level_%d' % (prefix, days)
        #         print('Adding `%s`...' % target_field)
        #         data: pd.Series = ori[ori_field].fillna(0)
        #         if 'days' in prefix:
        #             res_df[target_field] = data
        #         else:
        #             if is_train:
        #                 res_df[target_field], cut_bins[ori_field] = pd.qcut(
        #                     data, q=bins, labels=False, retbins=True, precision=1, duplicates='drop')
        #             else:
        #                 tmp_bins = cut_bins[ori_field]
        #                 tmp_bins[0], tmp_bins[-1] = data.min(), data.max()  # 确保包含所有元素
        #                 res_df[target_field] = pd.cut(
        #                     data, bins=tmp_bins, labels=False, precision=1, include_lowest=True)
        #
        # if is_train:
        #     # Persist cut_bins for predict data.
        #     if not os.path.exists(os.path.dirname(bins_path)):
        #         os.makedirs(bins_path)
        #     pickle.dump(cut_bins, open(bins_path, mode='wb'))

        return res_df


if __name__ == '__main__':
    _base_dir = os.path.dirname(__file__)
    _data_dir = os.path.join(_base_dir, 'data')
    _preprocess_dir = os.path.join(_base_dir, 'data', 'preprocess')
    _predict_dir = os.path.join(_base_dir, 'data', 'predict_data')

    _city_distance = pd.read_csv(os.path.join(_data_dir, 'city_distance.csv'))

    # For training.
    _driver_action = pd.read_csv(os.path.join(_data_dir, 'driver_action.csv'))
    _driver_portrait = pd.read_csv(os.path.join(_data_dir, 'driver_portrait.csv'))
    _cargo_record = pd.read_csv(os.path.join(_data_dir, 'cargo_record.csv'))

    # # make training original DataFrame
    _ori_df = make_original_df(_driver_action, _driver_portrait, _cargo_record, _city_distance)
    _ori_df.to_csv(os.path.join(_preprocess_dir, 'train_original.csv'), index=False)

    # make training features DataFrame
    _ori_df = pd.read_csv(os.path.join(_preprocess_dir, 'train_original.csv'))
    _features_df = make_features_df(_ori_df, is_train=True,
                                    bins_path=os.path.join(_preprocess_dir, 'cut_bins.obj'))
    _features_df.to_csv(os.path.join(_preprocess_dir, 'train_features.csv'), index=False)

    # For prediction.
    _predict = pd.read_csv(os.path.join(_predict_dir, 'predict.csv'), header=None)
    _predict = pd.DataFrame({'user_id': _predict.iloc[:, 0],
                             'id': _predict.iloc[:, 1]})
    _driver_portrait = pd.read_csv(os.path.join(_predict_dir, 'driver_portrait.csv'))
    _cargo_record = pd.read_csv(os.path.join(_predict_dir, 'cargo_record.csv'))

    # make predict original DataFrame
    _ori_df = make_original_df(_predict, _driver_portrait, _cargo_record, _city_distance, is_train=False)
    _ori_df.to_csv(os.path.join(_preprocess_dir, 'predict_original.csv'), index=False)

    # make predict features DataFrame
    _ori_df = pd.read_csv(os.path.join(_preprocess_dir, 'predict_original.csv'))
    _features_df = make_features_df(_ori_df, is_train=False,
                                    bins_path=os.path.join(_preprocess_dir, 'cut_bins.obj'))
    _features_df.to_csv(os.path.join(_preprocess_dir, 'predict_features.csv'), index=False)
