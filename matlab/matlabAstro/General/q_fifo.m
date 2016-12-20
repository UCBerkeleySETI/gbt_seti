function [Q,Out]=q_fifo(Q,Command,PushElement)
%--------------------------------------------------------------------------
% q_fifo function                                                  General
% Description: Implementation of a first-in-first-out queue (FIFO).
%              The queue is implemented using a cell array and each
%              element in the queue can be of any matlab type.
% Input  : - Cell array of all elements in the queue.
%          - Command type:
%            'push'   - Push a new element into the queue.
%            'get'    - Get an element from the queue.
%            'n'      - Return the number of elements in the queue.
%            'delete' - Delete all elements in queue.
%            If not given then return only the queue.
%          - Optional element to push.
% Output : - The queue.
%          - Output (if options 'get' | 'N' where used).
% Tested : Matlab 7.6
%     By : Eran O. Ofek                    Mar 2009
%    URL : http://weizmann.ac.il/home/eofek/matlab/
% Example: [Q,Out]=q_fifo({},'push',1);   % create a queue and push element
%          [Q,Out]=q_fifo(Q,'push',2);    % push another element
% Reliable: 1
%--------------------------------------------------------------------------

Nq = length(Q);
if (nargin==1),
   Out = [];
else
  switch lower(Command)
   case 'push'
      Q{Nq+1} = PushElement;   % add new element to the end of queue
      Out     = [];
   case 'get'
      Out     = Q{1};
      Q       = Q(2:Nq); % remove the element from queue
   case 'n'
      % number of elements in queue
      Out     = Nq;
   case 'delete'
      Q       = {};   % delete all elements
      Out     = [];
   otherwise
     error('Unknown Command option');
  end
end
