"""Local mock check; no blpapi import/session/network."""
from types import SimpleNamespace
from datetime import date
from bloomberg_earnings_econ_download_reviewed import BloombergClient,complete_history,chunks


class Message:
    def __init__(self,cid,value):self.cid,self.value=cid,value
    def correlationIds(self):return [self.cid]
    def messageType(self):return 'HistoricalDataResponse'
    def asElement(self):return self
    def toPy(self):return {'value':self.value}


class Event(list):
    def eventType(self):return 5


class Session:
    def __init__(self):self.events=iter([Event([Message(99,'stale')]),Event([Message(1,'current')])])
    def getService(self,_):return self
    def createRequest(self,_):return SimpleNamespace(fromPy=lambda _:None)
    def sendRequest(self,_,correlationId):assert correlationId==1
    def nextEvent(self,_):return next(self.events)


def run():
    client=BloombergClient.__new__(BloombergClient)
    client.blpapi=SimpleNamespace(CorrelationId=lambda x:x,Event=SimpleNamespace(RESPONSE=5))
    client.session=Session();client.next_request=0;client.timeout=1
    assert client.send({})==[{'value':'current'}]
    row={'security':'NDX Index','field_id':'BE008','value':100}
    assert complete_history([row],['NDX Index'],['BE008'])
    assert not complete_history([row],['NDX Index','SPX Index'],['BE008'])
    assert not complete_history([dict(row,value=float('nan'))],['NDX Index'],['BE008'])
    periods=chunks(date(2010,1,1),date(2026,9,8))
    assert len(periods)*4==16
    assert all((b[0]-a[1]).days==1 for a,b in zip(periods,periods[1:]))
    print('PASS: stale response excluded, incomplete/nonfinite data rejected, 16-request plan')


if __name__=='__main__':run()
