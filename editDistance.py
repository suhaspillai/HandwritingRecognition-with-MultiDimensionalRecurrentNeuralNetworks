
import numpy as  np
import pdb
class Check_edit_distance():
    '''
    Class implements Edit distance for computing CER i.e (Character Error Rate)
    '''

    def __init__(self):
        pass

    def decode_best_path(self,probs, ref=None, blank=0):
        
        best_path = np.argmax(probs,axis=0).tolist()
      
        # Collapse phone string
        hyp = []
        for i,b in enumerate(best_path):
            # ignore blanks
            if b == blank:
                continue
            # ignore repeats
            elif i != 0 and  b == best_path[i-1]:
                continue
            else:
                hyp.append(b)
    
        # Optionally compute phone error rate to ground truth
        dist = 0
        if ref is not None:
            #ref = ref.tolist()
            dist,_,_,_,_ = self.edit_distance(ref,hyp)
        
        return hyp,dist
        
    def edit_distance(self,ref,hyp):
        """
        Edit distance between two sequences reference (ref) and hypothesis (hyp).
        Returns edit distance, number of insertions, deletions and substitutions to
        transform hyp to ref, and number of correct matches.
        """
        n = len(ref)
        m = len(hyp)

        ins = dels = subs = corr = 0
        
        D = np.zeros((n+1,m+1))

        D[:,0] = np.arange(n+1)
        D[0,:] = np.arange(m+1)

        for i in xrange(1,n+1):
            for j in xrange(1,m+1):
                if ref[i-1] == hyp[j-1]:
                    D[i,j] = D[i-1,j-1]
                else:
                    D[i,j] = min(D[i-1,j],D[i,j-1],D[i-1,j-1])+1

        i=n
        j=m
        while i>0 and j>0:
            if ref[i-1] == hyp[j-1]:
                corr += 1
            elif D[i-1,j] == D[i,j]-1:
                ins += 1
                j += 1
            elif D[i,j-1] == D[i,j]-1:
                dels += 1
                i += 1
            elif D[i-1,j-1] == D[i,j]-1:
                subs += 1
            i -= 1
            j -= 1

        ins += i
        dels += j

        return D[-1,-1],ins,dels,subs,corr

    def disp(self,ref,hyp):
        dist,ins,dels,subs,corr = self.edit_distance(ref,hyp)
        '''
        print "Reference : %s, Hypothesis : %s"%(''.join(ref),''.join(hyp))
        print "Distance : %d"%dist
        print "Ins : %d, Dels : %d, Subs : %d, Corr : %d"%(ins,dels,subs,corr)
        '''
        return dist,corr
 

