function Rotmat= rotationMatrix2d(theta)
%in the usual setting it should be the inverse but matlab work with the other orientation...
Rotmat=[cos(theta) sin(theta) ; - sin(theta) cos(theta) ];
end
