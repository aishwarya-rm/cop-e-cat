import sys
sys.path.append('../../')
from cop_e_cat.copecat import CopECat, CopECatParams
import numpy as np
from sklearn import preprocessing
import pickle
import os
import pandas as pd

state_feats = ['anchor_age', 'patientweight', 'gender',
       'cad', 'afib', 'chf', 'ckd', 'esrd', 'paralysis', 'parathyroid',
       'rhabdo', 'sarcoid', 'sepsis', 'expired', 'bpdia', 'bpsys', 'hr', 'rr',
       'spo2', 'temp', 'alt', 'aniongap', 'bun', 'cpk', 'ca', 'chloride',
       'creatinine', 'glucose', 'hgb', 'k', 'ldh', 'mg', 'na', 'p', 'wbc',
       'betablockers', 'ca-iv', 'ca-noniv', 'cablockers', 'dextrose',
       'hours-dextrose', 'fluids', 'insulin', 'k-iv', 'hours-k-iv', 'k-noniv',
       'hours-k-noniv', 'loopdiuretics', 'hours-loopdiuretics', 'mg-iv',
       'hours-mg-iv', 'mg-noniv', 'hours-mg-noniv', 'p-iv', 'hours-p-iv',
       'p-noniv', 'hours-p-noniv', 'pnutrition', 'ponutrition', 'tpnutrition',
       'vasopressors', 'hours-betablockers', 'hours-cablockers',
       'hours-insulin', 'hours-ca-noniv', 'hours-vasopressors']

def state_transformer(frames, params, states=None, feats=state_feats):
	print('#:', len(feats))
	transformer = params.output_dir + 'transformer.pkl'
	if states is None:
		states = np.vstack([np.array(frames.loc[i, feats]).astype(float) for i in range(len(frames))])
	if os.path.isfile(transformer):
		scaler = pickle.load(open(transformer, 'rb'))
	else:
		scaler = preprocessing.StandardScaler().fit(states)
		pickle.dump(scaler, open(params.output_dir + 'transformer.pkl', 'wb'))
	transformed_states = scaler.transform(states)
	return transformed_states


def state_invtransformer(tstates, params):
	transformer = params.output_dir + 'tranformer.pkl'
	scaler = pickle.load(open(transformer, 'rb'))
	states = scaler.inverse_transform(tstates)

	return states


def transform(state, params):
	transformer = params.output_dir + 'transformer.pkl'
	if os.path.isfile(transformer):
		scaler = pickle.load(open(transformer, 'rb'))
		return scaler.transform([state])[0]
	else:
		return state


def discretize(a, el='K'):
	if el == 'K':
		adict = {'none': 0, 'low2-iv': 0, 'low4-iv': 0, 'low6-iv': 0, 'high1-iv': 0,
				 'high2-iv': 0, 'high3-iv': 0, 'low-po': 0, 'med-po': 0, 'high-po': 0}
		# What is ivd, ivh?
		ivd = float(a[0])
		ivh = float(a[2])
		orald = float(a[1])
		# print(ivd, ivh, orald)

		if ivd > 0:
			if ivh == 0.0: ivh = 1.0
			rate = ivd / ivh

			if rate <= 10:
				if ivh <= 2:
					adict['low2-iv'] = 1
				elif ivh <= 4:
					adict['low4-iv'] = 1
				else:
					adict['low6-iv'] = 1

			if rate > 10:
				if ivh <= 1:
					adict['high1-iv'] = 1
				elif ivh <= 2:
					adict['high2-iv'] = 1
				else:
					adict['high3-iv'] = 1

		if orald > 0:
			if orald <= 20:
				adict['low-po'] = 1
			elif orald <= 40:
				adict['med-po'] = 1
			else:
				adict['high-po'] = 1


	elif el == 'Mg':

		adict = {'none': 0, 'low4-iv': 0, 'high1-iv': 0, 'high2-iv': 0, 'high3-iv': 0, 'low-po': 0, 'med-po': 0, 'high-po': 0}

		ivd = float(a[0])
		ivh = float(a[2])
		orald = float(a[1])
		# print(ivd, ivh, orald)

		if ivd > 0:
			if ivd > 4: ivd = 4
			if ivh == 0.0: ivh = 1.0
			rate = ivd / ivh
			if rate < 1: adict['low4-iv'] = 1
			if rate >= 1:
				if ivh <= 1:
					adict['high1-iv'] = 1
				elif ivh <= 2:
					adict['high2-iv'] = 1
				else:
					adict['high3-iv'] = 1

		if orald > 0:
			if orald < 400:
				adict['low-po'] = 1
			elif orald < 800:
				adict['med-po'] = 1
			else:
				adict['high-po'] = 1


	elif el == 'P':
		adict = {'none': 0, 'low2-iv': 0, 'high1-iv': 0, 'high3-iv': 0, 'low-po': 0, 'med-po': 0, 'high-po': 0}

		ivd = float(a[0])
		ivh = float(a[2])
		orald = float(a[1])
		# print(ivd, ivh, orald)

		if ivd > 0:
			if ivh == 0.0: ivh = 1.0
			rate = ivd / ivh

			if rate <= 1: adict['low2-iv'] = 1
			if rate > 1:
				if ivh < 6:
					adict['high1-iv'] = 1
				else:
					adict['high3-iv'] = 1

		if orald > 0:
			if orald < 250:
				adict['low-po'] = 1
			elif orald < 500:
				adict['med-po'] = 1
			else:
				adict['high-po'] = 1

	da = list(adict.values())
	if sum(da) == 0: da[0] = 1

	return da


def reward(s, a, ns, w=np.array([1, 1, 1, 1, 1]) / 5., el='K'):
	rdict = {'cost-iv': 0, 'cost-po': 0, 'high': 0, 'low': 0, 'other': 0}

	if a[0] > 0: rdict['cost-iv'] -= 1
	if a[1] > 0: rdict['cost-po'] -= 1

	if el == 'K': rdict['high'], rdict['low'] = sigmoid(ns[0], el=el)
	if el == 'Mg': rdict['high'], rdict['low'] = sigmoid(ns[1], el=el)
	if el == 'P': rdict['high'], rdict['low'] = sigmoid(ns[2], el=el)
	# What does this mean? And-ing a set of floats
	#     if el == 'K':
	#         print(str(s[30]))
	#         print(str(s[31]))
	#         print(str(s[32]))
	#         rdict['other'] = -1 * (s[30] & s[31] & s[32])

	phi = np.array(list(rdict.values()))
	r = np.dot(phi, w)

	return phi, r


def sigmoid(x, el='K'):
	minmax = {'K': [3.5, 4.5], 'Mg': [1.5, 2.5], 'P': [2.5, 4.5]}
	lmin, lmax = minmax[el]

	if x < lmin:
		z = 1 / (1 + np.exp(-3.5 * (x - (lmin - 1)))) - 1
		return (0, z)
	elif x > lmax:
		z = - 1 / (1 + np.exp(-3.5 * (x - (lmax + 1))))
		return (z, 0)
	else:
		z = 0
		return (z, z)


def generate_samples(vnum, trainFrames, el='K'):
	frame = trainFrames[trainFrames.hadm_id == vnum]
	all_st = []
	all_nst = []
	all_a = []
	all_phi = []
	all_r = []

	for i in frame.index[:-1]:

		s = list(frame.loc[i, state_feats])
		st = transform(s)
		all_st.append(st)
		if el == 'K':
			a = list(frame.loc[i + 1, ['k-iv', 'k-noniv', 'hours-k-iv']])
		# a = list(frame.loc[i + 1, ['k-iv', 'k-noniv']])
		elif el == 'Mg':
			a = list(frame.loc[i + 1, ['mg-iv', 'mg-noniv', 'hours-mg-iv']])
		# a = list(frame.loc[i + 1, ['mg-iv', 'mg-noniv']])
		elif el == 'P':
			a = list(frame.loc[i + 1, ['p-iv', 'p-noniv', 'hours-p-iv']])
		# a = list(frame.loc[i + 1, ['p-iv', 'p-noniv']])
		da = discretize(a, el=el)
		all_a.append(da)
		ns = list(frame.loc[i + 1, state_feats])
		nst = transform(ns)
		all_nst.append(nst)
		phi, r = reward(s, a, ns, el=el)
		all_phi.append(phi)
		all_r.append(r)
	# print('s:', s, '\n\na:', a, '\n\nns', ns, '\n\nr', phi, r)

	return (all_st, all_a, all_nst, all_phi, all_r)


def combine(ent):
	return np.concatenate(np.array(ent))


def get_tuples(frames, params, filename='tuples.pkl', el='K'):
	transition_tuples = {'s': [], 'a': [], 'ns': [], 'phi': [], 'r': [], 'vnum': []}

	if el == 'K':
		visits = frames[(frames['k-iv'] != 0) | (frames['k-noniv'] != 0)].hadm_id.unique()
	elif el == 'Mg':
		visits = frames[(frames['mg-iv'] != 0) | (frames['mg-noniv'] != 0)].hadm_id.unique()
	elif el == 'P':
		visits = frames[(frames['p-iv'] != 0) | (frames['p-noniv'] != 0)].hadm_id.unique()
	else:
		visits = frames.visit_num.unique()

	for vnum in visits:
		if len(frames[frames.hadm_id == vnum]) > 1:
			s, a, ns, phi, r = generate_samples(vnum, frames, el)
			transition_tuples['s'].append(np.array(s))
			transition_tuples['a'].append(np.array(a))
			transition_tuples['ns'].append(ns)
			transition_tuples['phi'].append(phi)
			transition_tuples['r'].append(r)
			transition_tuples['vnum'].append(np.repeat(vnum, len(r)))

	for k in transition_tuples.keys():
		transition_tuples[k] = combine(transition_tuples[k])

	pickle.dump(transition_tuples, open(params.output_dir + filename, 'wb'))

	return transition_tuples

if __name__ == '__main__':
	params = CopECatParams('params.json')
	print("Generating state spaces")
	copecat = CopECat(params)
	copecat.generate_state_spaces()
	allFrames = pd.read_csv(params.output_dir + 'allFrames.csv')

	print('Total number of processed adms =', len(allFrames.hadm_id.unique()),'; number of transitions =', len(allFrames))
	allFrames = allFrames.sort_values(by=['hadm_id', 'timestamp'])
    
	trainFrames = allFrames[:int(len(allFrames)*0.75)].reset_index()
	testFrames = allFrames[int(len(allFrames)*0.75):].reset_index()
	trainFrames.to_csv(params.output_dir+'trainFrames.csv', index=False)
	testFrames.to_csv(params.output_dir+'testFrames.csv', index=False)
	import ipdb; ipdb.set_trace()
	print('Transformer')
	ts = state_transformer(trainFrames, params)
	visits = allFrames.hadm_id.unique()

	print('Training set:')
	print('Total number of processed adms =',len(trainFrames.hadm_id.unique()),'; number of transitions =',len(trainFrames))
	# if False:
	print ('Potassium Cohort size')
	print('Number administered iv K =', len(allFrames[allFrames['k-iv'] != 0].hadm_id.unique()))
	print('Number administered oral K =', len(allFrames[allFrames['k-noniv'] != 0].hadm_id.unique()))
	print('Get tuples')
	get_tuples(trainFrames, params, filename='trainKtuples.pkl', el='K')
	get_tuples(testFrames, params, filename='testKtuples.pkl', el='K')